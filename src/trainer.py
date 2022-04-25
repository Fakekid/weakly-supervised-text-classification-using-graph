# coding:utf8

import os
import numpy as np
import torch
from torch import nn
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
from .loss import KLLoss

from tqdm import tqdm
from prettytable import PrettyTable

from .modules import ClsModel
from transformers import AutoTokenizer

from ..evaluate import multi_cls_metrics
from ..dataset_po import ClsDataset
from ..modules.optimizer import build_optimizer


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

    def __init__(self, ptm_name, **kwargs):
        for k, v in kwargs.items():
            print(f'设置 {k}={v}')
            exec(f'self.{k} = {v}')
            self.kl_loss = KLLoss(reduction='sum')
            self.model = ClsModel.from_pretrained(ptm_name)

    def prepare_soft_label_data(self, rank, model, idx):
        target_num = min(self.world_size * self.train_batch_size * self.update_interval * self.accum_steps,
                         len(self.train_data["input_ids"]))
        if idx + target_num >= len(self.train_data["input_ids"]):
            select_idx = torch.cat((torch.arange(idx, len(self.train_data["input_ids"])),
                                    torch.arange(idx + target_num - len(self.train_data["input_ids"]))))
        else:
            select_idx = torch.arange(idx, idx + target_num)
        assert len(select_idx) == target_num
        idx = (idx + len(select_idx)) % len(self.train_data["input_ids"])
        select_dataset = {"input_ids": self.train_data["input_ids"][select_idx],
                          "attention_masks": self.train_data["attention_masks"][select_idx]}
        dataset_loader = self.make_dataloader(rank, select_dataset, self.eval_batch_size)
        input_ids, input_mask, preds = self.inference(model, dataset_loader, rank, return_type="data")
        gather_input_ids = [torch.ones_like(input_ids) for _ in range(self.world_size)]
        gather_input_mask = [torch.ones_like(input_mask) for _ in range(self.world_size)]
        gather_preds = [torch.ones_like(preds) for _ in range(self.world_size)]
        dist.all_gather(gather_input_ids, input_ids)
        dist.all_gather(gather_input_mask, input_mask)
        dist.all_gather(gather_preds, preds)
        input_ids = torch.cat(gather_input_ids, dim=0).cpu()
        input_mask = torch.cat(gather_input_mask, dim=0).cpu()
        all_preds = torch.cat(gather_preds, dim=0).cpu()
        weight = all_preds ** 2 / torch.sum(all_preds, dim=0)
        target_dist = (weight.t() / torch.sum(weight, dim=1)).t()
        all_target_pred = target_dist.argmax(dim=-1)
        agree = (all_preds.argmax(dim=-1) == all_target_pred).int().sum().item() / len(all_target_pred)
        self_train_dict = {"input_ids": input_ids, "attention_masks": input_mask, "labels": target_dist}
        return self_train_dict, idx, agree

    def train(self, ptm_name, num_labels, data, output_path, x_col='text', y_col='label',
              batch_size=128, max_seq_len=128, model_type='cls', device='cuda',
              weight_decay=0.01, learning_rate=1e-5, warmup_ratio=0.1, label_balance=None,
              val_data=None, epoch=10):
        """
        训练模型
        """
        tokenizer = AutoTokenizer.from_pretrained(ptm_name)
        dataset = ClsDataset(data, tokenizer=tokenizer, max_seq_len=max_seq_len, x_col=x_col, y_col=y_col)
        loader = DataLoader(dataset, batch_size=batch_size)

        loader_valid = None
        if val_data is not None:
            dataset_valid = ClsDataset(val_data, tokenizer=tokenizer, max_seq_len=max_seq_len, x_col=x_col, y_col=y_col)
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
        bar = tqdm(range(1, epoch + 1))
        for e in bar:
            model.train()

            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                plabels = batch['plabels'].to(device)
                labels = batch['labels'].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)

                loss = self.kl_loss(output, plabels)

                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_steps += 1

                # TODO: calculate

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

    def infer_for_soft_label(self, dataset_loader):
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

        return all_input_ids, all_input_mask, all_preds
