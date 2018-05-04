# -*- encoding: utf-8 -*-
from pybloom import BloomFilter, ScalableBloomFilter
import os, sys, argparse
from itertools import combinations, chain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import mmh3
from pybloom import BloomFilter
from scipy import spatial
from collections import Counter

arc_param = argparse.ArgumentParser()
arc_param.add_argument('-d', '--datapath', type=str, default='./data/samples.txt')
arc_param.add_argument('-s', '--stopwords', default='./data/stopwords.txt')
arc_param.add_argument('-e', '--expected', type=int, default=28)
arc_param.add_argument('--dim', type=int, default=200, help='intermediate feature dim')
arc_param.add_argument('-k', '--kterm', type=int, default=3, help='The k-term size')
arc_param.add_argument('--ratio', type=float, default=0.7, help='the classifier ratio')
params = arc_param.parse_args()

# load data
with open(params.datapath, 'r') as filin:
    corpus = [lin.strip().split('|||||') for lin in filin.readlines()]
tfidf_extractor = TfidfVectorizer(stop_words=open(params.stopwords, 'r').read().replace('\n', ' ').split())
pure_corpus = [lin[1] for lin in corpus]
title_corpus = [lin[0] for lin in corpus]

corpus_vec = tfidf_extractor.fit_transform(pure_corpus)
corpus_vec = corpus_vec.toarray()

# reduce dimesion of features
mat = np.random.randn(corpus_vec.shape[1], params.dim)
reduce_tfidf = np.array([np.matmul(itm, mat) for itm in corpus_vec])

# save features
if not os.path.exists('./tmp'):
    os.mkdir('tmp')
with open('./tmp/'+str(params.dim)+'_dim_features.txt', 'w') as fout:
    for idx, vector in zip(corpus, reduce_tfidf):
        vector_ = str(np.array(vector, np.float16).tolist()).replace('\n', '').strip('[').strip(']')
        fout.write(idx[0]+', '+vector_+'\n')
feature_file = './tmp/'+str(params.dim)+'_dim_features.txt'

corpus_token = [lin.split(' ') for lin in pure_corpus]
corpus_set = [set(lin) for lin in corpus_token]


def findsubsets(S):
    return chain.from_iterable(combinations(S, m) for m in range(1, params.kterm+1))


# find the k-term set of each document
corpus_sub_sets = [findsubsets(itm) for itm in corpus_set]

# novelty score
f = BloomFilter(capacity=5000000, error_rate=0.01)
com = 0
nec = 0
novelity = []
naieve_novlity = []
for docu, snt in zip(corpus_sub_sets, corpus_token):
    sntnovl = 0
    totaln = 0
    sntlen = len(snt)
    for itm in docu:
        com += 1
        # Hn <-- H_(n-1) + cn (if cn not in H_(n-1))
        if f.add(mmh3.hash(''.join(sorted(itm)))):
            #print(sorted(itm))
            nec += 1
            # In paper there will be an alpha value to different k-term length for scores
            # (but the author not explain how to)
            sntnovl += len(itm)
            totaln += 1
    novelity.append(float(sntnovl)*sntlen/totaln/10)
    naieve_novlity.append(float(sntnovl)/totaln)
        # print(mmh3.hash(str(sorted(itm))),'\t', len(itm), '\t', sorted(itm) if f.add(mmh3.hash(str(itm))) else '')
print('Total n-term number:', com, '\nRepreated Bloom element number', nec)
clsb = sorted(novelity, reverse=True)
# clsb = sorted(naieve_novlity, reverse=True)
tw_idx = [novelity.index(ele) for ele in clsb]
# tw_idx = [naieve_novlity.index(ele) for ele in clsb]
cluster_dict = {}.fromkeys(range(params.expected))

for tid, ttil in enumerate(tw_idx[:params.expected]):
    cluster_dict[tid] = [ttil]
    print(title_corpus[ttil])


def compute_dis(feature):
    total_fea = [np.mean(reduce_tfidf[itm], axis=0) for itm in cluster_dict.values()]
    cosine_dis = [1-spatial.distance.cosine(cfea, feature) for cfea in total_fea]
    return np.argmax(cosine_dis)

for tid in tw_idx[params.expected:]:
    new_tfea = reduce_tfidf[tid]
    c_id = compute_dis(new_tfea)
    print(title_corpus[tid], '\t --> \t', cluster_dict[c_id])
    cluster_dict[c_id].append(tid)

for ky in cluster_dict.keys():
    cluster_dict[ky] = set([title_corpus[wid] for wid in cluster_dict[ky]])

if not os.path.exists('./results'):
    os.mkdir('./results')

with open('./results/'+'%d_clusters.txt'%params.expected, 'w') as filout:
    for key in cluster_dict.keys():
        filout.write('\t'.join(cluster_dict[key])+'\n')

# evaluation
result_file = "./results/%d_clusters.txt"%params.expected
lins = [lin.strip().split('\t') for lin in open(result_file, 'r').readlines() if lin.strip() != '']


def get_true_total(lisin):
    ref_dic = Counter()
    sta = {}
    for itm_ in lins:
        refined_list = [itm.strip('\'').split('_')[0] for itm in itm_]
        ref_dic.update(refined_list)
    for k, v in ref_dic.most_common():
        sta[k.strip('sample')] = v
    return sta


true_num = get_true_total(lins)


def lst2counter(list_):
    refined_list = [itm.strip('\'').split('_')[0] for itm in list_]
    cnt = Counter()
    cnt.update(refined_list)
    total = len(list_)
    key, value = cnt.most_common(1)[0]
    key = key.strip('sample')
    # return (key, total, value)
    return key, float(value)/total, float(value)/true_num[key]


dicts = {}
precdict = {}
for itm in lins:
    key_, recal, pre = lst2counter(itm)
    if key_ in dicts.keys():
        dicts[key_].append(recal)
        precdict[key_] = precdict[key_] + pre
    else:
        dicts.update({key_: [recal]})
        precdict.update({key_: pre})
for k, v in dicts.items():
    # print('cluster '+k+':\t'+str(np.mean(dicts[k])))
    dicts[k] = np.mean(dicts[k])

dicts = {}
precdict = {}
for itm in lins:
    key_, recal, pre = lst2counter(itm)
    if key_ in dicts.keys():
        dicts[key_].append(recal)
        precdict[key_] = precdict[key_] + pre
    else:
        dicts.update({key_: [recal]})
        precdict.update({key_: pre})

for k, v in dicts.items():
    # print('cluster ' + k + ':\t' + str(np.mean(dicts[k])))
    dicts[k] = np.mean(dicts[k])

precision = sum(dicts.values()) / len(true_num.keys())
recall = sum(precdict.values()) / len(true_num.keys())
F1 = 2 * recall * precision / (precision + recall)
#print(recall, precision, F1)
print('To_predict_num:\t {}'.format(len(true_num.keys())))
print('Predicted_num:\t {}'.format(len(dicts)))
print('Set cluster num:\t {}'.format(params.expected))
print('Precision: {:0.4f}\t Recall: {:0.4f}\t F1: {:0.4f}'.format(precision*100, recall*100, F1*100))
