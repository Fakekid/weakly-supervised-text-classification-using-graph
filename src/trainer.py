# coding:utf8

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask
)
from loss import KLLoss

from tqdm import tqdm
from prettytable import PrettyTable

from modules import ClsModel
from transformers import AutoTokenizer

from evaluate import multi_cls_metrics
from dataset import ClsDataset
from optimizer import build_optimizer
import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class PretrainTrainer:
    """

    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            print(f'设置 {k}={v}')
            exec(f'self.{k} = {v}')

    def train(self, model_name, data, mlm_rate=0.15, batch_size=None, epochs=None, learning_rate=None, save_steps=1000,
              save_total_limit=5, max_seq_len=512, x_col='text', output_path=None,
              prediction_loss_only=True, seed=None, logging_steps=100):
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            prediction_loss_only=prediction_loss_only,
            seed=seed
        )

        tokenizer = build_optimizer(model_name)
        model_config = BertConfig.from_pretrained(model_name)

        if model_name.find('wwm') == -1:
            data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer,
                                                         mlm=True,
                                                         mlm_probability=mlm_rate)
        else:

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                            mlm=True,
                                                            mlm_probability=mlm_rate)

        dataset = ClsDataset(data, tokenizer=tokenizer, max_seq_len=max_seq_len, x_col=x_col)

        model = BertForMaskedLM.from_pretrained(model_name, config=model_config)

        trainer = Trainer(
            # model=torch.nn.parallel.DistributedDataParallel(model),
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )

        trainer.train()
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)


class FinetuneTrainer:
    """

    """

    def __init__(self, ptm_name, num_labels, **kwargs):
        for k, v in kwargs.items():
            print(f'设置 {k}={v}')
            exec(f'self.{k} = {v}')
        self.model = ClsModel.from_pretrained(ptm_name)
        self.model._init_vars(num_labels)
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(ptm_name)

    def calc_q(self, data):
        """

        """
        plabel = torch.FloatTensor(data['plabel'].str.split(',').apply(lambda x: [float(i) for i in x]).tolist())
        plabel = torch.softmax(plabel, dim=-1)
        f = torch.sum(plabel, dim=0)
        # plabel[m, n], f[n]
        q = (plabel ** 2 / f) / torch.sum((plabel ** 2 / f))

        self.q = q.to('cuda')
        logging.info(f'mean for class {torch.mean(q, dim=0)}')
        logging.info(f'min for class {torch.min(q, dim=0)}')
        logging.info(f'max for class {torch.max(q, dim=0)}')

    def calc_loss(self, logits, idx):
        """

        """
        logging.info('calc kl_div...')
        logits = nn.Softmax(dim=-1)(logits)
        logging.info(f'logits {logits.size()}, q_batch {self.q[idx * self.batch_size: (idx + 1) * self.batch_size].softmax(dim=-1).size()}')
        loss = F.kl_div(logits.softmax(dim=-1).log(),
                        self.q[idx * self.batch_size: (idx + 1) * self.batch_size].softmax(dim=-1),
                        reduction='batchmean')

        return loss

    def train(self, data, output_path,
              batch_size=128, max_seq_len=128, device='cuda',
              weight_decay=0.01, learning_rate=1e-5, warmup_ratio=0.1,
              val_data=None, epoch=10):
        """
        训练模型
        """
        logging.info('calc q...')
        self.calc_q(data)

        logging.info('start train')
        self.batch_size = batch_size
        num_labels = self.num_labels
        tokenizer = self.tokenizer
        dataset = ClsDataset(data, tokenizer=tokenizer, max_seq_len=max_seq_len)
        loader = DataLoader(dataset, batch_size=batch_size)

        loader_valid = None
        if val_data is not None:
            dataset_valid = ClsDataset(val_data, tokenizer=tokenizer, max_seq_len=max_seq_len)
            loader_valid = DataLoader(dataset_valid, batch_size=batch_size)

        # 释放显存占用
        torch.cuda.empty_cache()

        train_num = len(loader)
        train_steps = int(train_num * epoch / batch_size) + 1

        model = self.model

        optimizer, scheduler = build_optimizer(model, train_steps, learning_rate=learning_rate,
                                               weight_decay=weight_decay, warmup_ratio=warmup_ratio)
        model.to(device)

        total_loss = 0.0
        global_steps = 0
        global_acc = 0
        # bar = tqdm(range(1, epoch + 1))
        for e in range(epoch):
            logging.info(f'current epoch {e + 1}')
            model.train()
            bar = tqdm(loader)
            for idx, batch in enumerate(bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)

                # TODO: calculate self-label loss
                loss = self.calc_loss(output, idx)

                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_steps += 1

                acc = torch.argmax(torch.softmax(output, dim=-1), dim=-1) == labels

                acc = acc.type(torch.float)
                acc = torch.mean(acc)

                bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                    global_steps, round(acc.item(), 4), round(loss.item(), 4), round(scheduler.get_lr()[0], 7)))

            if loader_valid is not None:
                # 每个epoch结束进行验证集评估并保存模型
                labels = None
                outputs = None
                for batch in loader_valid:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    labels_batch = batch['labels'].to(device)

                    output = model(input_ids=input_ids, labels=labels_batch,
                                   token_type_ids=token_type_ids, attention_mask=attention_mask)

                    output = output[1].cpu().detach().numpy()
                    labels_batch = labels_batch.cpu().detach().numpy()

                    output = np.argmax(output, axis=-1)
                    if labels is None:
                        labels = labels_batch
                        outputs = output
                    else:
                        labels = np.concatenate([labels, labels_batch], axis=0)
                        outputs = np.concatenate([outputs, output], axis=0)

                labels = np.reshape(labels, [-1])
                outputs = np.reshape(outputs, [-1])

                met = multi_cls_metrics(labels, outputs, need_sparse=False, num_labels=num_labels)
                acc = met['acc']

                table = PrettyTable(['global_steps',
                                     'loss',
                                     'lr',
                                     'acc'])
                table.add_row([global_steps + 1,
                               total_loss / (global_steps + 1),
                               scheduler.get_lr()[0],
                               round(acc, 4)])
                logging.info(table)

                if global_acc < acc:
                    output_path_ = output_path
                    if not os.path.exists(output_path_):
                        os.mkdir(output_path_)
                    model_save_path = os.path.join(output_path_, 'finetune_model_best_acc')
                    model_to_save = model.module if hasattr(model, 'module') else model

                    model_to_save.save_pretrained(model_save_path)

            else:
                output_path_ = output_path
                if not os.path.exists(output_path_):
                    os.mkdir(output_path_)
                model_save_path = os.path.join(output_path_, f'finetune_model_{e}')
                model_to_save = model.module if hasattr(model, 'module') else model

                model_to_save.save_pretrained(model_save_path)

    def infer(self, dataset_loader):
        """
        inference function for building soft-label
        needs to predict all documents' class-prob
        """
        all_input_ids = []
        all_input_mask = []
        all_preds = []
        for batch in dataset_loader:
            with torch.no_grad():
                input_ids = batch[0].to('cuda')
                input_mask = batch[1].to('cuda')
                logits = self.model(input_ids,
                                    token_type_ids=None,
                                    attention_mask=input_mask)
                logits = logits[:, 0, :]

                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_preds.append(nn.Softmax(dim=-1)(logits))

        return all_preds
