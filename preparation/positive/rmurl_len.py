import os, sys, re
import codecs
ll = codecs.open(sys.argv[1], 'r', 'utf-8').readlines()

res = re.compile(r'http(\S)*', re.S)

with codecs.open(sys.argv[1]+'.rmurlen.txt', 'w', 'utf-8') as filout:
    for lin in ll:
        lin = res.sub('', lin)
        lin_ = lin.strip().split()
        if len(lin_)<8:
            continue
        else:
            filout.write(lin)
