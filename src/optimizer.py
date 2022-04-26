# coding:utf8

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def build_optimizer(model, train_steps, learning_rate, weight_decay, warmup_ratio):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * warmup_ratio,
    #                                  t_total=train_steps)

    return optimizer
