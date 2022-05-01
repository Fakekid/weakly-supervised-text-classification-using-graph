# coding: utf8
import torch
from torch.utils.data import Dataset


class CLSDataset(Dataset):
    def __init__(self, df, tokenizer, max_seq_len):
        self.df = df
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.text = df['text'].values
        if 'label' in df.columns:
            self.label = df['label'].values
        if 'plabel' in df.columns:
            self.plabel = df['plabel'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length'
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return_dict = {}
        if 'label' in self.df.columns:
            label = self.label[index]
            return_dict['labels'] = torch.tensor(label, dtype=torch.long)
        if 'plabel' in self.df.columns:
            plabel = [float(i) for i in self.plabel[index].split(',')]
            return_dict['plabels'] = torch.tensor(plabel, dtype=torch.float)

        return_dict['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        return_dict['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)

        return return_dict


class MCPDataset(Dataset):
    def __init__(self, df, tokenizer, max_seq_len):
        self.df = df
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.text = df['text'].values
        if 'label' in df.columns:
            self.label = df['label'].values
        if 'plabel' in df.columns:
            self.plabel = df['plabel'].values
            self.mask_place = df['mask_place'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length'
        )

        return_dict = {}

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return_dict['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        return_dict['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)

        if 'label' in self.df.columns:
            label = self.label[index]
            return_dict['labels'] = torch.tensor(label, dtype=torch.long)

        if 'plabel' in self.df.columns:
            plabel = [float(i) for i in self.plabel[index].split(',')]
            return_dict['plabels'] = torch.tensor(plabel, dtype=torch.float)

            mask_place = [int(i) for i in self.mask_place[index].split(',')]
            return_dict['mask_place'] = torch.tensor(mask_place, dtype=torch.long)

        return return_dict
