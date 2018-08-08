#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import pandas as pd

import sklearn

import time, datetime
import itertools
from joblib import Parallel, delayed
import pickle

parser = argparse.ArgumentParser(description='convert csv to pickle')
parser.add_argument('-d', '--data', type=str, default='alpha', help='data name')
parsed = parser.parse_args(sys.argv[1:])

data_name = parsed.data
print(data_name)

network_df = pd.read_csv('../rev2data/%s/%s_network.csv' %(data_name, data_name), header=None, names=['src', 'dest', 'rating', 'timestamp'], parse_dates=[3], infer_datetime_format=True)
user_list = ['u' + str(u) for u in network_df['src'].tolist()]
gt_df = pd.read_csv('../rev2data/%s/%s_gt.csv' %(data_name, data_name), header=None, names=['id', 'label'])
gt = dict([('u'+str(x[0]), x[1]) for x in zip(gt_df['id'], gt_df['label'])])

def compute_score_avg(k=0, n=0, ind=0):
    user_list = ['u' + str(u) for u in pd.read_csv('../rev2data/%s/%s_network.csv' %(data_name, data_name), header=None, names=['src', 'dest', 'rating', 'timestamp'])['src'].tolist()]
    ret = {}
    cnt = 0
    missing_files = []
    for c1, c2, c3, c4, c5, c6, c7 in itertools.product(range(1, 3), range(1, 3), range(1, 3), range(1, 3), range(1, 3), range(1, 3), range(0, 0)):
        file_path = '../rev2res/%s/%s-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d.csv' %(data_name, data_name, c1, c2, c3, c4, c5, c6, c7, k, n, ind)
        if not os.path.exists(file_path): missing_files += [file_path]; continue
        df = pd.read_csv(file_path, header=None)
        d = dict(zip(df[0], df[1]))
        if len(ret) == 0: ret = d
        assert len(ret) == len(d)
        s = {k:ret[k]+d[k] for k in ret}
        ret = s
        # if cnt % 3**5 == 0: print('    ', cnt)
        cnt += 1
    print('%d,%d,%d cnt %d, missing %d' %(k, n, ind, cnt, len(missing_files)))

    user_score = ret
    u_list = [u for u in user_score if u in gt or u not in user_list]
    ytrue = [1 if u in gt and gt[u] == 1 else 0 for u in u_list]
    yscore = [user_score[u] for u in u_list]
    return {'ulist': u_list, 'ytrue': ytrue, 'yscore': yscore}

results = Parallel(n_jobs=-1, verbose=3)(delayed(compute_score_avg)(k, n, ind) for k, n, ind in itertools.product(range(11), range(11), range(20)))

results_dict = dict(zip(itertools.product(range(11), range(11), range(20)), results))

with open('../rev2res/%s.pkl' %data_name, 'wb') as f:
    pickle.dump(results_dict, f)
