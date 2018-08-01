import os
import sys
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
# plt.style.use('seaborn-paper')
import seaborn as sns
matplotlib.rc('text', usetex=True)

import time, datetime
import itertools

def compute_score_avg(data_name='alpha', k=0, n=0):
    user_list = ['u' + str(u) for u in pd.read_csv('../rev2data/%s/%s_network.csv' %(data_name, data_name), header=None, names=['src', 'dest', 'rating', 'timestamp'])['src'].tolist()]
    ret = {}
    cnt = 0
    missing_files = []
    for c1, c2, c3, c4, c5, c6, c7, ind in itertools.product(range(1, 3), range(1, 3), range(1, 3), range(1, 3), range(1, 3), range(1, 3), range(1, 3), range(20)):
        if c5 == 0 and c6 == 0: continue
        file_path = '../rev2res/%s/alpha-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d.csv' %(data_name, c1, c2, c3, c4, c5, c6, c7, k, n, ind)
        if not os.path.exists(file_path): missing_files += [file_path]; continue
        df = pd.read_csv(file_path, header=None)
        d = dict(zip(df[0], df[1]))
        if len(ret) == 0: ret = d
        assert len(ret) == len(d)
        s = {k:ret[k]+d[k]  for k in ret}
        ret = s
        # if cnt % 3**5 == 0: print('    ', cnt)
        cnt += 1
    print('total', cnt)
    print('missing files', len(missing_files))
    return ret

for k, n in itertools.product(range(11), range(11)):
    print(k, n)
    u_score = compute_score_avg('alpha', k, n)
    score_df = pd.DataFrame.from_dict(data = u_score, orient='index')
    score_df.to_csv('../rev2res/%s-res/%d-%d.csv' %('alpha', k, n), header=False)

