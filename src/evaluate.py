# coding:utf8

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def bin_cls_metrics(preds, labels):
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    pred_cate = preds > 0.5

    acc = np.mean(pred_cate == labels)

    f1 = f1_score(labels, pred_cate, zero_division=0)

    p = precision_score(labels, pred_cate, zero_division=0)
    r = recall_score(labels, pred_cate, zero_division=0)

    try:
        auc = roc_auc_score(y_true=labels, y_score=preds)
    except Exception as e:
        auc = -1
    return auc, acc, f1, p, r


def multi_cls_metrics(labels, logits, average='micro', return_auc_when_multi_cls=None, need_sparse=True,
                      has_mask=False, num_labels=None):
    if num_labels is None:
        all_labels = None
    else:
        all_labels = [list(range(1, num_labels))]

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
    return {'acc': acc}

    # f1 = f1_score(labels, preds, average=average, zero_division=0, labels=all_labels)
    #
    # p = precision_score(labels, preds, average=average, zero_division=0, labels=all_labels)
    # r = recall_score(labels, preds, average=average, zero_division=0, labels=all_labels)
    #
    # if logits.shape[-1] > 2:
    #     if return_auc_when_multi_cls is not None:
    #         return {'auc': return_auc_when_multi_cls, 'acc': acc, 'f1': f1, 'p': p, 'r': r}
    #     else:
    #         return {'acc': acc, 'f1': f1, 'p': p, 'r': r}
    # else:
    #     auc = roc_auc_score(labels, logits[:, 1], average=average)
    # return {'auc': auc, 'acc': acc, 'f1': f1, 'p': p, 'r': r}


evaluate = multi_cls_metrics


# def evaluate(data, model):
#     """
#     线下评测函数
#     """
#     Y_true, Y_pred = [], []
#     for x_true, y_true in data:
#         y_pred = model.predict(x_true)[:, 0, 5:7]  # 看第0个位置的5、6两个token的概率值，不关注其它token其它类别的概率值
#         y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
#         y_true = y_true[:, 0] - 5
#         Y_pred.extend(y_pred)
#         Y_true.extend(y_true)
#     return roc_auc_score(Y_true, Y_pred)


# def multi_cls_metrics(logits, labels, average='macro', return_auc_when_multi_cls=None, need_sparse=True,
#                       has_mask=False):
#     # all_labels = [x for x in range(logits.shape[1])]
#     if need_sparse:
#         preds = np.argmax(logits, axis=-1).reshape(-1)
#     else:
#         preds = logits.reshape(-1)
#     labels = labels.reshape(-1)
#
#     if has_mask:
#         mask = labels > 0
#         labels = labels[mask]
#         preds = preds[mask]
#
#     acc = np.mean(preds == labels)
#     # acc = np.sum((preds == labels) * mask) / np.sum(mask)
#
#     f1 = f1_score(labels, preds, average=average, zero_division=0)
#
#     # p = precision_score(labels, preds, labels=all_labels[1:], average=average, zero_division=0)
#     # r = recall_score(labels, preds, labels=all_labels[1:], average=average, zero_division=0)
#
#     p = precision_score(labels, preds, average=average, zero_division=0)
#     r = recall_score(labels, preds, average=average, zero_division=0)
#
#     if logits.shape[-1] > 2:
#         if return_auc_when_multi_cls is not None:
#             return {'auc': return_auc_when_multi_cls, 'acc': acc, 'f1': f1, 'p': p, 'r': r}
#         else:
#             return {'acc': acc, 'f1': f1, 'p': p, 'r': r}
#     else:
#         auc = roc_auc_score(labels, logits[:, 1], average=average)
#     return {'auc': auc, 'acc': acc, 'f1': f1, 'p': p, 'r': r}



