# coding:utf8

import os
import numpy as np
from sklearn.utils import shuffle
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

from modules import OurModel
from transformers import AutoTokenizer

from evaluate import multi_cls_metrics
from dataset import CLSDataset, MCPDataset
from optimizer import build_optimizer
import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def assessment(loader, model, device, num_labels):
    logging.info('now valid...')
    # 每个epoch结束进行验证集评估并保存模型
    labels = None
    outputs = None
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        output = output.cpu().detach().numpy()
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

        return acc


class MLMTrainer:
    """

    """

    def __init__(self, ptm_name, num_labels, **kwargs):
        for k, v in kwargs.items():
            print(f'设置 {k}={v}')
            exec(f'self.{k} = {v}')
        self.model = OurModel.from_pretrained(ptm_name, output_attentions=False, output_hidden_states=False,
                                              num_labels=num_labels)
        self.model._init_vars(num_labels)
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(ptm_name)

    def calc_loss(self, logits, targets, mask=None):
        if mask is None:
            loss = F.kl_div(logits.softmax(dim=-1).log(),
                            targets.softmax(dim=-1),
                            reduction='batchmean')
        else:
            logits_ = torch.einsum('ijk,ij->ijk', logits.softmax(dim=-1).log(), mask)
            targets_ = torch.einsum('ijk,ij->ijk', targets.softmax(dim=-1), mask)
            loss = F.kl_div(logits_,
                            targets_,
                            reduction='batchmean')
        # logging.info(f'loss {loss}')
        return loss

    def train(self, data, output_path, batch_size=128, max_seq_len=128, device='cuda', weight_decay=0.01,
              learning_rate=1e-5, warmup_ratio=0.1, val_data=None, epoch=10):
        logging.info('start train')
        self.batch_size = batch_size
        num_labels = self.num_labels
        tokenizer = self.tokenizer

        loader_valid = None
        if val_data is not None:
            dataset_valid = MCPDataset(val_data, tokenizer=tokenizer, max_seq_len=max_seq_len)
            loader_valid = DataLoader(dataset_valid, batch_size=batch_size)

        # 释放显存占用
        torch.cuda.empty_cache()

        train_num = len(data) // batch_size + 1
        train_steps = int(train_num * epoch / batch_size) + 1

        model = self.model

        optimizer = build_optimizer(model, train_steps, learning_rate=learning_rate,
                                    weight_decay=weight_decay, warmup_ratio=warmup_ratio)
        model.to(device)

        total_loss = 0.0
        global_steps = 0
        global_acc = 0
        for e in range(epoch):
            logging.info(f'current epoch {e + 1}')
            model.train()

            # shuffle data for each epoch
            data = shuffle(data)

            # logging.info('calc q...')
            # self.calc_q(data)

            dataset = CLSDataset(data, tokenizer=tokenizer, max_seq_len=max_seq_len)
            loader = DataLoader(dataset, batch_size=batch_size)

            accu_acc = 0
            bar = tqdm(loader)
            all_preds = None
            all_labels = None
            for idx, batch in enumerate(bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                plabels = batch['plabels'].to(device)
                mask_place = batch['mask_place'].to(device)
                labels = batch['labels'].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask, pred_mode='mlm')

                # TODO: calculate self-label loss

                loss = self.calc_loss(output, plabels, mask=mask_place)

                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                # scheduler.step()
                model.zero_grad()
                global_steps += 1

                preds_batch = torch.argmax(torch.softmax(output, dim=-1), dim=-1).cpu().detach().numpy()
                labels_batch = labels.cpu().detach().numpy()
                if all_labels is None:
                    all_labels = labels_batch
                    all_preds = preds_batch
                else:
                    all_labels = np.concatenate([all_labels, labels_batch], axis=0)
                    all_preds = np.concatenate([all_preds, preds_batch], axis=0)

                acc = np.mean((all_labels == all_preds).astype('float32'))

                # acc = torch.argmax(torch.softmax(output, dim=-1), dim=-1) == labels
                # acc = acc.type(torch.float)
                # acc = torch.mean(acc)
                # accu_acc = (idx * batch_size * accu_acc + batch_size * acc) / ((idx + 1) * batch_size)

                bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                    global_steps, round(acc.item(), 4), round(loss.item(), 4), round(learning_rate * 1e6, 2)))
                # bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                #     global_steps, round(acc.item(), 4), round(loss.item(), 4), round(scheduler.get_lr()[0] * 1e6, 2)))

            if loader_valid is not None:
                acc = assessment(loader_valid, model, device, num_labels)

                table = PrettyTable(['global_steps',
                                     'loss',
                                     'acc'])
                table.add_row([global_steps + 1,
                               total_loss / (global_steps + 1),
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


class CLSTrainer:
    """

    """

    def __init__(self, ptm_name, num_labels, **kwargs):
        for k, v in kwargs.items():
            print(f'设置 {k}={v}')
            exec(f'self.{k} = {v}')
        self.model = OurModel.from_pretrained(ptm_name)
        self.model._init_vars(num_labels)
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(ptm_name)

    def calc_q(self, data):
        """

        """
        plabel = torch.FloatTensor(data['plabel'].str.split(',').apply(lambda x: [float(i) for i in x]).tolist())
        plabel = torch.softmax(plabel, dim=-1) + 1e-7
        f = torch.sum(plabel, dim=0)
        # plabel[m, n], f[n], b[m]
        a = plabel ** 2 / f
        b = torch.sum((plabel ** 2 / f), dim=1)
        q = a.transpose(1, 0) / b
        q = q.transpose(1, 0)

        self.q = q.to('cuda')
        logging.info(f'mean for class {torch.mean(q, dim=0)}')
        logging.info(f'min for class {torch.min(q, dim=0)}')
        logging.info(f'max for class {torch.max(q, dim=0)}')

    def calc_soft_label_loss(self, logits, idx):
        """

        """
        loss = F.kl_div(logits.softmax(dim=-1),
                        self.q[idx * self.batch_size: (idx + 1) * self.batch_size].softmax(dim=-1),
                        reduction='batchmean')
        # logging.info(f'loss {loss}')
        return loss

    def calc_loss(self, logits, targets):
        loss = F.kl_div(logits.softmax(dim=-1).log(),
                        targets.softmax(dim=-1),
                        reduction='batchmean')
        # logging.info(f'loss {loss}')
        return loss

    def train(self, data, output_path,
              batch_size=128, max_seq_len=128, device='cuda',
              weight_decay=0.01, learning_rate=1e-5, warmup_ratio=0.1,
              val_data=None, epoch=10):
        """
        训练模型
        """
        logging.info('start train')
        self.batch_size = batch_size
        num_labels = self.num_labels
        tokenizer = self.tokenizer

        loader_valid = None
        if val_data is not None:
            dataset_valid = CLSDataset(val_data, tokenizer=tokenizer, max_seq_len=max_seq_len)
            loader_valid = DataLoader(dataset_valid, batch_size=batch_size)

        # 释放显存占用
        torch.cuda.empty_cache()

        train_num = len(data) // batch_size + 1
        train_steps = int(train_num * epoch / batch_size) + 1

        model = self.model

        optimizer = build_optimizer(model, train_steps, learning_rate=learning_rate,
                                    weight_decay=weight_decay, warmup_ratio=warmup_ratio)
        model.to(device)

        total_loss = 0.0
        global_steps = 0
        global_acc = 0
        for e in range(epoch):
            logging.info(f'current epoch {e + 1}')
            model.train()

            # shuffle data for each epoch
            data = shuffle(data)

            # logging.info('calc q...')
            # self.calc_q(data)

            dataset = CLSDataset(data, tokenizer=tokenizer, max_seq_len=max_seq_len)
            loader = DataLoader(dataset, batch_size=batch_size)

            accu_acc = 0
            bar = tqdm(loader)
            all_preds = None
            all_labels = None
            for idx, batch in enumerate(bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                plabels = batch['plabels'].to(device)
                labels = batch['labels'].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)

                # TODO: calculate self-label loss
                # loss = self.calc_loss(output, idx)
                loss = self.calc_loss(output, plabels)

                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                # scheduler.step()
                model.zero_grad()
                global_steps += 1

                preds_batch = torch.argmax(torch.softmax(output, dim=-1), dim=-1).cpu().detach().numpy()
                labels_batch = labels.cpu().detach().numpy()
                if all_labels is None:
                    all_labels = labels_batch
                    all_preds = preds_batch
                else:
                    all_labels = np.concatenate([all_labels, labels_batch], axis=0)
                    all_preds = np.concatenate([all_preds, preds_batch], axis=0)

                acc = np.mean((all_labels == all_preds).astype('float32'))

                # acc = torch.argmax(torch.softmax(output, dim=-1), dim=-1) == labels
                # acc = acc.type(torch.float)
                # acc = torch.mean(acc)
                # accu_acc = (idx * batch_size * accu_acc + batch_size * acc) / ((idx + 1) * batch_size)

                bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                    global_steps, round(acc.item(), 4), round(loss.item(), 4), round(learning_rate * 1e6, 2)))
                # bar.set_description('step:{} acc:{} loss:{} lr:{}'.format(
                #     global_steps, round(acc.item(), 4), round(loss.item(), 4), round(scheduler.get_lr()[0] * 1e6, 2)))

            if loader_valid is not None:
                acc = assessment(loader_valid, model, device, num_labels)
                # logging.info('now valid...')
                # # 每个epoch结束进行验证集评估并保存模型
                # labels = None
                # outputs = None
                # for batch in tqdm(loader_valid):
                #     input_ids = batch['input_ids'].to(device)
                #     attention_mask = batch['attention_mask'].to(device)
                #     labels_batch = batch['labels'].to(device)
                #
                #     output = model(input_ids=input_ids, attention_mask=attention_mask)
                #
                #     output = output.cpu().detach().numpy()
                #     labels_batch = labels_batch.cpu().detach().numpy()
                #
                #     output = np.argmax(output, axis=-1)
                #     if labels is None:
                #         labels = labels_batch
                #         outputs = output
                #     else:
                #         labels = np.concatenate([labels, labels_batch], axis=0)
                #         outputs = np.concatenate([outputs, output], axis=0)
                #
                # labels = np.reshape(labels, [-1])
                # outputs = np.reshape(outputs, [-1])
                #
                # met = multi_cls_metrics(labels, outputs, need_sparse=False, num_labels=num_labels)
                # acc = met['acc']

                table = PrettyTable(['global_steps',
                                     'loss',
                                     'acc'])
                table.add_row([global_steps + 1,
                               total_loss / (global_steps + 1),
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
