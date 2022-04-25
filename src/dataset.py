# coding: utf8
import torch
from torch.utils.data import Dataset


class ClsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df['text'].values
        if 'label' in df.columns:
            self.label = df['label'].values
            self.plabel = df['plabel'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if 'label' in self.df.columns:
            label = self.label[index]
            plabel = [float(i) for i in self.plabel[index].split(',')]

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long),
                'plabels': torch.tensor(plabel, dtype=torch.float)
            }
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
