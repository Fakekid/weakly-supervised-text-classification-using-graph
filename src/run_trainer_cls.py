# coding:utf-8

import time
import warnings

import pandas as pd

from trainer import FinetuneTrainer


def main():
    with open('conf_cls.txt', 'r', encoding='utf8') as fin:
        c = fin.readlines()
    config = eval(''.join(c))

    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print("\n>> loading model from :{}".format(config['ptm_name']))

    dataset = pd.read_csv(config['train_data_path'])

    if config['adv'] == '':
        print('\n>> start normal training ...')
    elif config['adv'] == 'fgm':
        print('\n>> start fgm training ...')
    elif config['adv'] == 'pgd':
        print('\n>> start pgd training ...')

    ft = FinetuneTrainer(ptm_name=config['ptm_name'], num_labels=config['num_labels'])
    ft.train(dataset, epoch=config['num_epochs'], output_path=config['output_path'], batch_size=config['batch_size'],
             learning_rate=config['learning_rate'])

    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at:{}".format(localtime_end))


if __name__ == '__main__':
    main()
