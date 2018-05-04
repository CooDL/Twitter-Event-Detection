# coding=utf-8
import gensim
import os,sys
from collections import Counter
from LocalitySensitiveHashing import *
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

argp = argparse.ArgumentParser()
argp.add_argument('-d', '--datapath', default='./data/samples.txt')
argp.add_argument('-s', '--stopwords', default='./data/stopwords.txt')
argp.add_argument('--dim', type=int, default=200)
argp.add_argument('-e', '--expected', type=int, default=28)
params = argp.parse_args()

print('''\n------------------Usage-----------------------:

python xxx.py -d data_path -s stopwords_path --dim feature_dim -e expected_cluster_num

## feature_dim:  the transformed feature dim
## expected_cluster_num: the cluster num you want to get
## NOTE: the cluster result will be saved under ./results/
''')

# generate tfidf features
tfidf_extractor = TfidfVectorizer(stop_words=open(params.stopwords, 'r').read().replace('\n', ' ').split())
corpus = [lin.strip().split('|||||') for lin in open(params.datapath, 'r').readlines()]
pure_corpus = [lin[1] for lin in corpus]
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

# clustering
lsh = LocalitySensitiveHashing(
                   datafile=feature_file,
                   dim=200,
                   r=50,
                   b=100,
                   expected_num_of_clusters=params.expected,)
lsh.get_data_from_csv()
lsh.initialize_hash_store()
lsh.hash_all_data()
similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence(similarity_groups)
merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_sample_based(coalesced_similarity_groups)

# saving results
if not os.path.exists('./results'):
    os.mkdir('results')
lsh.write_clusters_to_file(merged_similarity_groups, "./results/%d_clusters.txt"%params.expected)

# evaluation
result_file = "./results/%d_clusters.txt"%params.expected
lins = [lin.strip().strip('set()[]').split(', ') for lin in
        open(result_file, 'r').readlines() if lin.strip() != '']


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
