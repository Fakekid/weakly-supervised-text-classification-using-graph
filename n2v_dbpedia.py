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
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet

import dgl.function as fn

import pickle as pkl


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


with open('datasets/dbpedia/label_names.txt') as fin:
    label_names = [item.replace('\n', '') for item in fin.readlines()]


with open('datasets/dbpedia/train.txt') as fin:
    text = [item.replace('\n', '') for item in fin.readlines()]
    
smp_idxs = np.random.randint(0, 5, len(text))
text = [text[i] for i in range(len(smp_idxs)) if smp_idxs[i] == 0]

# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# text = [tokenizer.tokenize(item) for item in tqdm(text)]

# text = list(chain.from_iterable(text))


def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

text = [remove_urls(item) for item in tqdm(text)]

text = [re.sub(r'[^\w\s]',' ', item) for item in tqdm(text)]
# text = [item.replace('\\', '').replace('<b>', '').replace('</b>', '').replace('#', '') for item in text]
text[:10]

text = [item for item in text if 30 < len(item) < 1000]
print(f'corpus length {len(text)}')

with open('data/en_names.txt', 'r') as fin:
    en_names = list(map(lambda x: x.replace('\n', ''), fin.readlines()))
en_names = [item for item in en_names if item not in label_names]

weeks = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday', 
         'January','February','March','April','May','June','July','August','September','October','November','December']

address = ["Birmingham", "Montgomery", "Mobile", "Anniston", "Gadsden", "Phoenix", "Scottsdale", "Tempe", "Buckeye", "Chandler", "ElDorado", "Jonesboro", "PaineBluff", "LittleRock", "Fayetteville", "FortSmith", "MileHouse", "Kelowna", "PrinceGeorge", "Modesto", "LosAngeles", "Monterey", "SanJose", "SanFrancisco", "Oakland", "Berkeley", "WalnutCreek", "Alturas", "Chico", "Reading", "Fresno", "Norwalk", "Downey", "LongBeach", "SanDiego", "Burbank", "Glendale", "SouthPasadena", "Arcadia", "LosAltos", "PaloAlto", "SouthSanFrancisco", "Eureka", "SantaRosa", "Sonoma", "Anaheim", "Barstow", "PalmSprings", "Bakersfield", "SantaBarbara", "Ventura", "NorthHollywood", "SanFernando", "Salinas", "SolanaBeach", "Riverside", "SanBernardino", "Sacramento", "Pleasanton", "Irvine", "Laguna", "Niguel", "ColoradoSprings", "Pueblo", "Boulder", "Denver", "Aspen", "FortCollins", "GrandJunction", "Bridgeport", "NewHaven", "Hartford", "KeyWest", "Kissimmee", "Gainesville", "Orlando", "BocaRaton", "Sebastian", "WestPalmBeach", "Clearwater", "NorthMiami", "St.Petersburg", "Tampa", "PanamaCity", "Pensacola", "Tallahassee", "AvonPark", "Jacksonville", "FortMyers", "Naples", "Sarasota", "FortLauderdale", "Americus", "Bainbridge", "Valdosta", "WarnerRobins", "Atlanta", "Alpharetta", "Augusta", "Rome", "Atlantasuburbs", "Brunswick", "Macon", "Savannah", "Waycross", "Champaign-Urbana", "Peoria", "RockIsland", "Alton", "Cairo", "EastSt.Louis", "Aurora", "Naperville", "OakBrookTerrace", "Chicagosuburbs", "Joliet", "LaSalle", "Rockford", "Chicago", "Evanston", "Waukegan", "FortWayne", "Gary", "Hammond", "SouthBend", "Indianapolis", "Kokomo", "Evansville", "TerreHaute", "CedarRapids", "Davenport", "Dubuque", "Waterloo", "Ames", "DesMoines", "FortDodge", "Creston", "MasonCity", "CouncilBluffs", "SiouxCity", "Coolidge", "DodgeCity", "Hutchinson", "Wichita", "Topeka", "Manhattan", "Colby", "Goodland", "Lawrence", "Salina", "Hopkinsville", "Owensboro", "Frankfort", "Louisville", "Morehead", "Lexington", "Jellico", "Kensee", "Lot", "Oneida", "Saxton", "BatonRouge", "NewRoads", "Shreveport", "LakeCharles", "Lafayette", "Houma", "NewOrleans", "Cumberland", "Frederick", "Hagerstown", "Annapolis", "Baltimore", "Rockville", "Salisbury", "Pittsfield", "Hyannis", "NewBedford", "Worcester", "Boston", "Norwood", "Weymouth", "Fitchburg", "Methuen", "Peabody", "TraverseCity", "Ludington", "Muskegon", "Detroit", "Lansing", "MountPleasant", "BattleCreek", "Kalamazoo", "AnnArbor", "Monroe", "Flint", "NorthernDetroitsuburbs", "Marquette", "SaultSte.Marie", "Duluth", "GrandRapids", "Moorhead", "StCloud", "Mankato", "Minneapolis", "SaintPaul", "RedWing", "MapleGrove", "Bloomington", "Gulfport", "Pascagoula", "Meriden", "Hattiesburg", "HollySprings", "Tupelo", "StCharles", "StLouis", "Union", "Joplin", "Nevada", "Hannibal", "JeffersonCity", "Independence", "KansasCity", "StJoseph", "GrandIsland", "NorthPlatte", "Scottsbluff", "Hastings", "Lincoln", "Omaha", "O'Neill", "LasVegas", "CarsonCity", "Reno", "Ely", "Hackensack", "Hoboken", "JerseyCity", "AtlanticCity", "Camden", "Trenton", "LongBranch", "NewBrunswick", "Vineland", "CherryHill", "Elizabeth", "Phillipsburg", "Washington", "Newark", "Paterson", "NewYorkCity", "Oswego", "Syracuse", "Utica", "Watertown", "Brentwood", "Hempstead", "Albany", "Gloversville", "Schenectady", "Troy", "Binghamton", "Elmira", "Endicott", "Ithaca", "LongIsland", "Manorville", "Buffalo", "NiagaraFalls", "Rochester", "Bronx", "Brooklyn", "Queens", "StatenIsland", "Flushing", "Poughkeepsie", "Peekskill", "WhitePlains", "Yonkers", "LabradorCity", "St.John's", "AtlanticBeach", "Hatteras", "Asheboro", "Thomasville", "Charlotte", "Concord", "Asheville", "Antioch", "Hickory", "Greensboro", "Winston-Salem", "Durham", "Raleigh", "Waterville", "Whitehorse", "Pangnirtung", "Cleveland", "Akron", "Canton", "Warren", "Youngstown", "BowlingGreen", "Findlay", "Lima", "Toledo", "Mentor", "Oberlin", "Westlake", "Cincinnati", "Middletown", "Cambridge", "Dayton", "Hillsboro", "Springfield", "Athens", "Columbus", "Lancaster", "Marietta", "Enid", "OklahomaCity", "Alva", "Ardmore", "Lawton", "McAlester", "Miami", "Muskogee", "Tulsa", "Toronto", "Guelph", "Kitchener", "London", "Windsor", "Barrie", "NorthBay", "SaultSteMarie", "Sudbury", "Dryden", "Kenora", "FortWilliam", "ThunderBay", "Cooksville", "Hamilton", "Mississauga", "Kingston", "Ottawa", "Astoria", "Beaverton", "Ashland", "Bend", "Corvallis", "Eugene", "Pendleton", "Salem", "Portland", "Philadelphia", "Pittsburgh", "Scranton", "Williamsport", "Philadelphiasuburbs", "Allentown", "Harrisburg", "Gettysburg", "NewCastle", "Latrobe", "Uniontown", "Altoona", "Erie", "Johnstown", "Chicoutimi", "Quebec", "Rimouski", "Montreal", "Lloydminster", "Regina", "Saskatoon", "RockHill", "Charleston", "HiltonHeadIsland", "MyrtleBeach", "Florence", "Anderson", "Greenville", "Spartanburg", "Bristol", "Chattanooga", "Nashville", "Knoxville", "Jackson", "Memphis", "UnionCity", "Columbia", "Manchester", "Cookeville", "SanAntonio", "Waco", "DeerPark", "CorpusChristi", "Victoria", "Beaumont", "Galveston", "Austin", "Bellaire", "Pasadena", "Amarillo", "Lubbock", "FortWorth", "DelRio", "Uvalde", "Houston", "Paris", "Sherman", "Texarkana", "Tyler", "Abilene", "ElPaso", "Huntsville", "Lufkin", "Denton", "WichitaFalls", "Brownsville", "McAllen", "Dallas", "Garland", "GrandPrairie", "Irving", "Plano", "Wharton", "SaintGeorge", "Richfield", "Blanding", "Moab", "SaltLakeCity", "Provo", "Ogden", "Blacksburg", "Roanoke", "Staunton", "Winchester", "Alexandria", "Arlington", "Fairfax", "Herndon", "Norfolk", "NewportNews", "Williamsburg", "Charlottesville", "Danville", "Richmond", "Seattle", "Auburn", "Kent", "Tacoma", "Bellingham", "Olympia", "Vancouver", "Bellevue", "Edmonds", "Everett", "Spokane", "WallaWalla", "Yakima", "WestBend", "Kenosha", "Milwaukee", "Racine", "Beloit", "LaCrosse", "Madison", "Platteville", "EauClaire", "Superior", "Wausau", "GreenBay"]

weeks = [item.lower() for item in weeks]
address = [item.lower() for item in address]

# nltk Tokenizer??????
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

tokenizer = RegexpTokenizer(r'\w+')

tweet = TweetTokenizer()

ps = nltk.PorterStemmer()

weeks = [item.lower() for item in weeks]
# weeks = []

stops = weeks + address + en_names

def is_en(s):
    for uchar in s:
        if ( uchar >= u'\u0041' and uchar <= u'\u005A' ) or ( uchar >= u'\u0061' and uchar <= u'\u007A'):
            continue
        else:
            return False
    return True


def get_nouns(x):
    nouns = []
    for t in tqdm(x):
#         tokens = word_tokenize(t)
        tokens = tweet.tokenize(t)
        pos_tags = nltk.pos_tag(tokens)

        noun = set()
        for word, pos in pos_tags:
            word = word.lower()
            if len(word) > 30 or len(word) < 3: continue
#             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' ) \
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'VBN' or pos == 'JJ' or pos == 'JJS') \
            and word not in stopwords.words('english') \
            and word not in stops \
            and is_en(word):
                noun.add(word)
#                 noun.add(wnl.lemmatize(word.lower()))
        nouns.append(list(noun))

    return nouns


nouns = process_data(text, get_nouns, num_workers=26)
print([' '.join(item) for item in nouns[:10]])

nouns = [item for item in nouns if 10 <= len(item)]
print('doc num', len(nouns))


def freq_filter(data, min_freq=1, return_cnt=False):
    """
    ???????????????
    """
    cnter = dict(Counter(list(chain.from_iterable(data))))
    cnter = {k: cnter[k] for k in sorted(cnter, key=lambda x: cnter[x], reverse=True) if cnter[k] > min_freq}
    
    if return_cnt:
        return set(cnter.keys()), cnter
    return set(cnter.keys())


all_words, cnter = freq_filter(nouns, min_freq=30, return_cnt=True)
print('word num', len(all_words))

with open('nouns.pkl', 'wb') as fout:
        pkl.dump(nouns, fout)
        
with open('word_counter_dic.pkl', 'wb') as fout:
        pkl.dump(cnter, fout)



w2i = {w: i for i, w in enumerate(all_words)}
i2w = {v: k for k, v in w2i.items()}


g_mat = np.zeros([len(all_words), len(all_words)])

graphs = []

for noun in tqdm(nouns):
    for u in noun:
        for v in noun:
            if not w2i.get(u) or not w2i.get(v): continue
#             if u == v: continue
            g_mat[w2i[u], w2i[v]] += 1
            g_mat[w2i[v], w2i[u]] += 1

print(f'graph size in mem: {g_mat.size / 1024 / 1024 / 1024}')

g_mat = np.log1p(g_mat)

g_nx = nx.from_numpy_array(g_mat)


print('graph sampling...')
from node2vec import Node2Vec
node2vec = Node2Vec(g_nx, dimensions=16, walk_length=16, num_walks=40, p=1., q=1.9)
# with open('node2vec-model/agnews_noun_n2v_p1.4_q1.2_wl16_nw20_dim16.pkl', 'wb') as fout:
#     pkl.dump(node2vec, fout)

print('w2v training...')
model = node2vec.fit(window=9, min_count=1)
model.save('node2vec-model/dbpedia_n2v_p1_q1_wl16_nw40_dim16.bin')


cate_sims = {}
for ln in label_names:
    print(ln)
    
    ws = []
    if w2i.get(ln) == None: continue
    for i, j in model.wv.most_similar(str(w2i[ln]), topn=10000):
        ws.append([i2w[int(i)], j])
        print(i2w[int(i)], j)
    print('-' * 40)
    cate_sims[ln] = ws


with open('dbpedia_n2v_cate_sims_p1q1_wsmp.pkl', 'wb') as fout:
	pkl.dump(cate_sims, fout)


print('all done.')

