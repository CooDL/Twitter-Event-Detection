#!/usr/bin/env python
#coding: utf-8

import sys
import codecs
import difflib as dfb
import re,os
from nltk.tokenize import TweetTokenizer
segtw = TweetTokenizer()

emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\uD83E[\uDD00-\uDDFF])|"
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83c[\udde0-\uddff])|"  # flags (iOS)
    u"([\u2934\u2935]\uFE0F?)|"
    u"([\u3030\u303D]\uFE0F?)|"
    u"([\u3297\u3299]\uFE0F?)|"
    u"([\u203C\u2049]\uFE0F?)|"
    u"([\u00A9\u00AE]\uFE0F?)|"
    u"([\u2122\u2139]\uFE0F?)|"
    u"(\uD83C\uDC04\uFE0F?)|"
    u"(\uD83C\uDCCF\uFE0F?)|"
    u"([\u0023\u002A\u0030-\u0039]\uFE0F?\u20E3)|"
    u"(\u24C2\uFE0F?|[\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55]\uFE0F?)|"
    u"([\u2600-\u26FF]\uFE0F?)|"
    u"([\u2700-\u27BF]\uFE0F?)"
    "+", flags=re.UNICODE)

url_pattern = re.compile(r'http(\S)*', re.S)
selected =[]
twdict = {}

def diffseq(seq1, seq2):

    cmpt = dfb.SequenceMatcher(None,seq1,seq2)
    locs = cmpt.find_longest_match(0, len(seq1), 0, len(seq2))
    sstm = seq1[locs[0]:locs[2]]
    if ' ' in sstm:
        # print 'shared seq', sstm
        depn = sstm.split()
        if len(depn) > 4:
            return True
        else:
            return False
    else:
        return False

def recurl(lista, listb): #lista non-repeat; listb sentence lib
    ''''''
    repeat = []
    for itmb in listb:
        includeflag = False
        for itma in lista:
            if diffseq(itma, itmb):
                includeflag = True
                repeat.append(itmb)
                break
        if not includeflag:
            lista.append(itmb)
    lista = set(lista)
    # print 'Before merge', len(listb)
    listb = set(listb)
    repeat = set(repeat)
    listb = listb - lista - repeat
    print "Step lista form listb",len(lista), len(listb)
    if len(listb) > 0:
        lista = list(lista)
        listb = list(listb)
        print '@@@@@@@@@@@@@@@@@@@@@@@@@ONE@@@@@MORE@@@@@@@@@@@REPEAT@@@@@@@@@@@@@\n'
        recurl(lista, listb)
    else:
        return list(lista)
    
def process(filname):
    INPUT_FILE = filname
    OUT_FILE = filname.rsplit('.', 1)[0]+'.rp.txt'
    with codecs.open(INPUT_FILE, 'r', 'utf-8') as infile, codecs.open (OUT_FILE, 'w', 'utf-8') as otfile:
        lines = [url_pattern.sub(r'', singline) for singline in infile.readlines()]
        truelines = [emoji_pattern.sub(r'', in_line.strip()) for in_line in lines if len(in_line.strip().split()) > 9]
        itmlines = list(set(truelines))
        truelines = [' '.join(segtw.tokenize(lin)) for lin in itmlines]
        selected = [truelines[0]]
        matching = truelines[1:]#[[itm] for itm in truelines[1:]]
        finaltxt = recurl(selected, matching)

        otfile.write('\n'.join(finaltxt))

allitm = os.listdir(sys.argv[1])
for itm in allitm:
    process(itm)
# with codecs.open(INPUT_FILE, 'r', 'utf-8') as infile:
#     itmlines = infile.readlines()
#     for itm in itmlines:         
#         itm = itm.strip() if len(itm.strip().split()) > 9 else continue              
#         if twdict.has_key(itm[-1]):
#             twdict[itm[-1]]= itm[0]
#             print 'Repeat with',itm[-1]
          
#         else:
#             twdict.update({itm[-1]:itm[0]})

# with codecs.open (OUT_FILE, 'w', 'utf-8') as otfile:
#     for sent in selected:
#         try:
#             otfile.write(twdict[sent]+'|||||'+sent+'\n')
#         except KeyError as e:
#             print e
