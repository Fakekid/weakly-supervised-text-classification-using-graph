# coding:utf8

import os
import numpy as np
import torch
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

from tqdm import tqdm
from prettytable import PrettyTable

from ..modules import BertForSequenceClassification, BertForTokenClassification
from ..modules.tokenizer import get_tokenizer
from ..evaluate import multi_cls_metrics
from ..dataset_po import ClsDataset
from ..modules.optimizer import build_optimizer


def build_finetune_model(ptm_name, num_labels, type='cls'):
    """

        Args:
            ptm_name: str value, pretrained model's name or local directory
            num_labels: int value, number of categories
            type: str value, downstream task type, default 'cls', current only support 'cls'、'tag'

        Returns:
            transformer model
        """
    if type == 'cls':
        model = BertForSequenceClassification.from_pretrained(ptm_name, num_labels=num_labels)
    elif type == 'tag':
        model = BertForTokenClassification.from_pretrained(ptm_name, num_labels=num_labels)
    else:
        raise ValueError('暂不支持cls和tag以外的模型')
    return model


def build_pretrain_model(model_name):
    """

    Args:
        model_name:

    Returns:

    """
    model_config = BertConfig.from_pretrained(model_name)

    model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                            config=model_config)
    return model


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

        tokenizer = get_tokenizer(model_name)
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

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            print(f'设置 {k}={v}')
            exec(f'self.{k} = {v}')

    def train(self, ptm_name, num_labels, data, output_path, x_col='text', y_col='label',
              batch_size=128, max_seq_len=128, model_type='cls', device='cuda',
              weight_decay=0.01, learning_rate=1e-5, warmup_ratio=0.1, label_balance=None,
              val_data=None):
        """
        训练模型
        """
        epoch = self.epoch

        tokenizer = get_tokenizer(ptm_name)
        dataset = ClsDataset(data, tokenizer=tokenizer, max_seq_len=max_seq_len, x_col=x_col, y_col=y_col)
        loader = DataLoader(dataset, batch_size=batch_size)

        need_valid = False
        if val_data is not None:
            need_valid = True
            dataset_valid = ClsDataset(val_data, tokenizer=tokenizer, max_seq_len=max_seq_len, x_col=x_col, y_col=y_col)
            loader_valid = DataLoader(dataset_valid, batch_size=batch_size)

        # 释放显存占用
        torch.cuda.empty_cache()

        train_num = len(loader)
        train_steps = int(train_num * epoch / batch_size) + 1

        model = build_finetune_model(ptm_name, num_labels, type=model_type)
        optimizer, scheduler = build_optimizer(model, train_steps, learning_rate=learning_rate,
                                               weight_decay=weight_decay, warmup_ratio=warmup_ratio)
        model.to(device)

        total_loss = 0.0
        global_steps = 0
        global_acc = 0
        bar = tqdm(range(1, epoch + 1))
        for e in bar:
            model.train()

            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               labels=labels, label_balance=label_balance, return_dict=True)
                loss = output['loss']

                optimizer.zero_grad()
                loss.backward()

                total_loss += loss.item()
                # cur_avg_loss += loss.item()

                optimizer.step()

                scheduler.step()
                model.zero_grad()

                global_steps += 1

                acc = torch.argmax(torch.softmax(output[1], dim=-1), dim=-1) == labels
                if model_type == 'cls':
                    acc = acc.type(torch.float)
                    acc = torch.mean(acc)
                else:
                    acc = acc.type(torch.float)
                    acc = torch.sum(acc * attention_mask)
                    acc = acc / torch.sum(attention_mask)
                bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                    global_steps, round(acc.item(), 4), round(loss.item(), 4), round(scheduler.get_lr()[0], 7)))

            if need_valid:
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

                if model_type == 'tag':
                    used_index = labels > 0
                    labels = labels[used_index]
                    outputs = outputs[used_index]

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
                print(table)

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
