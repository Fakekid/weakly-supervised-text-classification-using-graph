import networkx as nx
import matplotlib.pyplot as plt
import torch
import dgl
import os
from tqdm import tqdm
import jieba
import jieba.analyse
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from itertools import chain
import nltk
import re

from parallel_processor import process_data
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet

import dgl.function as fn

import pickle as pkl


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


with open('datasets/amazon/label_names.txt') as fin:
    label_names = [item.replace('\n', '') for item in fin.readlines()]


with open('datasets/amazon/train.txt') as fin:
    text = [item.replace('\n', '') for item in fin.readlines()]


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def sents_seg(text):
    r = [tokenizer.tokenize(item) for item in tqdm(text)]
    return r

text = process_data(text, sents_seg, num_workers=24)
text = text.tolist()
text = list(chain.from_iterable(text))
print('采样前', len(text))

smp_idx = np.random.randint(0, 100, len(text))
np.argwhere(smp_idx == 0)
text = [text[i[0]] for i in np.argwhere(smp_idx == 0)]
print('采样后', len(text))


def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

text = [remove_urls(item) for item in tqdm(text)]

text = [re.sub(r'[^\w\s]',' ', item) for item in tqdm(text)]
# text = [item.replace('\\', '').replace('<b>', '').replace('</b>', '').replace('#', '') for item in text]
text[:10]

text = [item for item in text if 5 < len(item) < 300]
print(f'corpus length {len(text)}')

weeks = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday', 
         'January','February','March','April','May','June','July','August','September','October','November','December']


weeks = [item.lower() for item in weeks]

# nltk Tokenizer方法
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

tokenizer = RegexpTokenizer(r'\w+')

tweet = TweetTokenizer()

ps = nltk.PorterStemmer()


weeks = [item.lower() for item in weeks]
# weeks = []


def get_nouns(x):
    nouns = []
    for t in tqdm(x):

        tokens = tweet.tokenize(t)
        pos_tags = nltk.pos_tag(tokens)

        noun = set()
        for word, pos in pos_tags:
            word = word.lower()
            if len(word) > 30 or len(word) < 3: continue
#             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' ) \
            if (pos == 'JJ' or pos == 'JJS' or pos == 'JJR') \
            and word not in stopwords.words('english') \
            and word not in weeks:
#                 noun.add(wnl.lemmatize(word, 'n'))
                noun.add(word)
#                 noun.add(wnl.lemmatize(word.lower()))
        nouns.append(list(noun))

    return nouns


nouns = process_data(text, get_nouns, num_workers=20)


def freq_filter(data, min_freq=1):
    """
    过滤低频词
    """
    cnter = dict(Counter(list(chain.from_iterable(data))))
    cnter = {k: cnter[k] for k in sorted(cnter, key=lambda x: cnter[x], reverse=True) if cnter[k] > min_freq}
    return set(cnter.keys())


all_words = freq_filter(nouns, min_freq=10)
print('词汇长度', len(all_words))



w2i = {w: i for i, w in enumerate(all_words)}
i2w = {v: k for k, v in w2i.items()}


g_mat = np.zeros([len(all_words), len(all_words)])

graphs = []

for noun in tqdm(nouns):
    for u in noun:
        for v in noun:
            if not w2i.get(u) or not w2i.get(v): continue
            if u == v: continue
            g_mat[w2i[u], w2i[v]] += 1
            g_mat[w2i[v], w2i[u]] += 1

print(f'graph size in mem: {g_mat.size / 1024 / 1024 / 1024}')

g_mat = np.log1p(g_mat)

g_nx = nx.from_numpy_array(g_mat)


print('graph sampling...')
from node2vec import Node2Vec
node2vec = Node2Vec(g_nx, dimensions=16, walk_length=16, num_walks=40, p=1.4, q=1.2)
# with open('node2vec-model/agnews_noun_n2v_p1.4_q1.2_wl16_nw20_dim16.pkl', 'wb') as fout:
#     pkl.dump(node2vec, fout)

print('w2v training...')
model = node2vec.fit(window=9, min_count=1)
model.save('node2vec-model/amazon_n2v_p1.4_q1.6_wl16_nw40_dim16_v2.bin')


cate_sims = {}
for ln in label_names:
    print(ln)
    
    ws = []
    for i, j in model.wv.most_similar(str(w2i[ln]), topn=100):
        ws.append([i2w[int(i)], j])
        print(i2w[int(i)], j)
    print('-' * 40)
    cate_sims[ln] = ws


with open('amazon_n2v_cate_sims.pkl', 'wb') as fout:
    pkl.dump(cate_sims, fout)


print('all done.')









