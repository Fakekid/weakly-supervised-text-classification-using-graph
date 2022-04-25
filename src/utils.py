from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def fetch_scheduler(optimizer, sche_name, **kwargs):
    scheduler = None

    if sche_name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs['T_max'],
            eta_min=kwargs['min_lr']
        )

    elif sche_name == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs['T_0'],
            eta_min=kwargs['min_lr']
        )
    elif sche_name == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            kwargs['weight_decay'],
            last_epoch=-1)

    return scheduler


def multi_cls_metrics(labels, logits, average='micro', return_auc_when_multi_cls=None, need_sparse=False,
                      has_mask=False, num_labels=None):
    if num_labels is None:
        all_labels = None
    else:
        all_labels = list(range(1, num_labels))

    if need_sparse:
        preds = np.argmax(logits, axis=-1).reshape(-1)
    else:
        preds = logits.reshape(-1)

    labels = labels.reshape(-1)

    if has_mask:
        mask = labels > 0
        labels = labels[mask]
        preds = preds[mask]

    acc = np.mean(preds == labels)

    f1 = f1_score(labels, preds, average=average, zero_division=0, labels=all_labels)

    p = precision_score(labels, preds, average=average, zero_division=0, labels=all_labels)
    r = recall_score(labels, preds, average=average, zero_division=0, labels=all_labels)

    if logits.shape[-1] > 2:
        if return_auc_when_multi_cls is not None:
            return {'auc': return_auc_when_multi_cls, 'acc': acc, 'f1': f1, 'p': p, 'r': r}
        else:
            return {'acc': acc, 'f1': f1, 'p': p, 'r': r}
    else:
        auc = roc_auc_score(labels, logits[:, 1], average=average)
    return {'auc': auc, 'acc': acc, 'f1': f1, 'p': p, 'r': r}


def read_dataset(data_path, tokenizer, max_seq_len, task, x_col, y_col, return_len=False, mode='train',
                 is_shuffle=True):
    seq_length = max_seq_len

    df = pd.read_csv(data_path, sep=',')
    length = df.shape[0]

    if mode == 'train':
        data = df[['text', 'label']].values
    else:
        data = df[['text']].values

    partial_processor = partial(data_processor, mode=mode, x_col=x_col, y_col=y_col, seq_length=seq_length,
                                tokenizer=tokenizer, task=task, is_shuffle=is_shuffle)

    df = process_data(data, partial_processor, num_workers=10)
    if mode == 'train':
        df = pd.DataFrame(df, columns=['input_ids', 'segment_ids', 'input_mask', 'label'])
    else:
        df = pd.DataFrame(df, columns=['input_ids', 'segment_ids', 'input_mask'])
    return df

    # for idx, row in tqdm(df.iterrows(), total=len(df)):
    #     sent = row[x_col]
    #     src = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(sent) + ['SEP'])
    #     seg = [0] * len(src)
    #     mask = [1] * len(src)
    #
    #     if len(src) > seq_length:
    #         src = src[: seq_length]
    #         seg = seg[: seq_length]
    #         mask = mask[: seq_length]
    #
    #     while len(src) < seq_length:
    #         src.append(0)
    #         seg.append(0)
    #         mask.append(0)
    #
    #     if mode == 'train':
    #         label = row[y_col]
    #         if task == 'tag' or task == 'tagging' or task == 'seq':
    #             label = label.split(' ')
    #             label = [int(1)] + list(map(int, label)) + [int(1)]
    #             if len(label) > seq_length:
    #                 label = label[:seq_length]
    #             while len(label) < seq_length:
    #                 label.append(0)
    #         else:
    #             label = int(label)
    #         dataset.append((src, seg, mask, label))
    #     else:
    #         dataset.append((src, seg, mask))
    #
    # if mode == 'train':
    #     data = pd.DataFrame(dataset, columns=['input_ids', 'segment_ids', 'input_mask', 'label'])
    # else:
    #     data = pd.DataFrame(dataset, columns=['input_ids', 'segment_ids', 'input_mask'])
    #
    # if is_shuffle:
    #     data = shuffle(data)
    #
    # if return_len:
    #     return data, length
    # return data


def build_tokenizer(path):
    if '/' in path:
        tokenizer_path = os.path.join(path, 'vocab.txt')
    else:
        tokenizer_path = path
    # tokenizer_path = config['model_path']
    # if config['tokenizer_fast']:
    #     tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    # else:
    #     tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    return tokenizer