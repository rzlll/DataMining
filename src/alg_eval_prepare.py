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
parser.add_argument('-a', '--alg', type=str, default='bad', help='alg name')
parsed = parser.parse_args(sys.argv[1:])

data_name = parsed.data
alg_name = parsed.alg
print(alg_name, data_name)

network_df = pd.read_csv('../rev2data/%s/%s_network.csv' %(data_name, data_name), header=None, names=['src', 'dest', 'rating', 'timestamp'], parse_dates=[3], infer_datetime_format=True)
user_list = ['u' + str(u) for u in network_df['src'].tolist()]
# gt_df = pd.read_csv('../rev2data/%s/%s_gt.csv' %(data_name, data_name), header=None, names=['id', 'label'])
# gt = dict([('u'+str(x[0]), x[1]) for x in zip(gt_df['id'], gt_df['label'])])

def parse_data(df):
    ret = []
    l0 = df[0].tolist()
    l1 = df[1].tolist()
    size = len(l0)
    for l in range(size):
        sline = l1[l].strip('()[]').split(',')
        ret.append([l0[l], sline[0]] + list(map(float, sline[1:])))
    if alg_name == 'bad':
        sortedret = sorted(ret, key=lambda x: x[3])
    else:
        sortedret = sorted(ret, key=lambda x: x[2])
    return pd.DataFrame(sortedret)

def compute_score(k=0, n=0, ind=0):
    results_df = pd.read_csv('../%sres/%s/%s-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind), header=None)
    if len(results_df.columns) < 3: results_df = parse_data(results_df)
    # if user is good in ground truth output 0
    # if user is fraudster in ground truth output 1
    # if user is sockpuppet output 2
    ytrue = [0 if t==1 else 1 for t in results_df[0].tolist()]
    ulist = [u if u in user_list else 's'+u[1:] for u in results_df[1].tolist()]
    yscore = results_df[2].tolist()
    return {'ulist': ulist, 'ytrue': ytrue, 'yscore': yscore}

results = Parallel(n_jobs=-1, verbose=3)(delayed(compute_score)(k, n, ind) for k, n, ind in itertools.product(range(11), range(11), range(50)))

results_dict = dict(zip(itertools.product(range(11), range(11), range(50)), results))

with open('../%sres/%s.pkl' %(alg_name, data_name), 'wb') as f:
    pickle.dump(results_dict, f)
