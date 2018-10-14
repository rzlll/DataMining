#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import pandas as pd

import sklearn

import time, datetime
import itertools
from joblib import Parallel, delayed
import pickle

parser = argparse.ArgumentParser(description='evaluate algorithms with data')
parser.add_argument('-d', '--data', type=str, default='alpha', choices=['alpha', 'amazon', 'epinions', 'otc'], help='data name')
parser.add_argument('-a', '--alg', type=str, default='bad', choices=['bn', 'feagle', 'fraudar', 'trust', 'rsd', 'bad'], help='alg name')
parser.add_argument('-n', '--ncores', type=int, default=-1, help='number of cores to use')
parsed = parser.parse_args(sys.argv[1:])

data_name = parsed.data
alg_name = parsed.alg
n_cores = parsed.ncores
print(alg_name, data_name)

network_df = pd.read_csv('../rev2data/%s/%s_network.csv' %(data_name, data_name), header=None, names=['src', 'dest', 'rating', 'timestamp'], parse_dates=[3], infer_datetime_format=True)
user_list = ['u' + str(u) for u in network_df['src'].tolist()]
gt_df = pd.read_csv('../rev2data/%s/%s_gt.csv' %(data_name, data_name), header=None, names=['id', 'label'])
gt = dict([('u'+str(x[0]), x[1]) for x in zip(gt_df['id'], gt_df['label'])])

def parse_data(df):
    ret = []
    l0 = df[0].tolist()
    l1 = df[1].tolist()
    size = len(l0)
    for l in range(size):
        sline = l1[l].strip('()[]').split(',')
        ret.append([l0[l], sline[0]] + list(map(float, sline[1:])))
    if alg_name == 'bad':
        sortedret = sorted(ret, key=lambda x: (x[2], x[3]))
    else:
        sortedret = sorted(ret, key=lambda x: x[2])
    return pd.DataFrame(sortedret)

def compute_score(k=0, n=0, ind=0):
    if alg_name == 'rev2':
        results_df = pd.read_csv('../res/%s/%s/%s-1-1-1-1-1-1-0-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind), header=None)
    else:
        results_df = pd.read_csv('../res/%s/%s/%s-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind), header=None)
    if len(results_df.columns) < 3: results_df = parse_data(results_df)
    # if user is good in ground truth output 0
    # if user is fraudster in ground truth output 1
    # if user is sockpuppet output 2
    ytrue = [0 if t==1 else 1 for t in results_df[0].tolist()]
    ulist = results_df[1].tolist()
    yscore = results_df[2].tolist()
    return {'ulist': ulist, 'ytrue': ytrue, 'yscore': yscore}

@numba.jit
def get_metrics(ytrue, yscore):
    '''get precision and recall at q percentile'''
    q = np.arange(2, 22, 2)/100

    assert len(ytrue) == len(yscore)
    size = len(ytrue)
    prec_dict = {}
    recl_dict = {}
    f1_dict = {}
    for qq in q:
        cut = qq*size
        ypred = (np.arange(size) < cut) * 1
        prec = sklearn.metrics.precision_score(y_pred=ypred, y_true=ytrue)
        recl = sklearn.metrics.recall_score(y_pred=ypred, y_true=ytrue)
        f1 = sklearn.metrics.f1_score(y_pred=ypred, y_true=ytrue)
        prec_dict[qq] = prec
        recl_dict[qq] = recl
        f1_dict[qq] = f1
    return prec_dict, recl_dict, f1_dict

def compute_metrics(res_dict, k, n, ind):
    ulist = np.array(res_dict[(k, n, ind)]['ulist'])
    yscore = np.array(res_dict[(k, n, ind)]['yscore'])
    ytrue = np.array(res_dict[(k, n, ind)]['ytrue'])
    prec_dict, recl_dict, f1_dict = get_metrics(ytrue, yscore)
    return {'prec': prec_dict, 'recl': recl_dict, 'f1': f1_dict}

n_range = list(range(0, 51, 5))
n_range[0] = 1

results_list = Parallel(n_jobs=n_cores, verbose=5)(delayed(compute_score)(k, n, ind) for k, n, ind in itertools.product(range(11), n_range, range(50)))

results_dict = dict(zip(itertools.product(range(11), n_range, range(50)), results_list))

metrics_list = Parallel(n_jobs=n_cores, verbose=5)(delayed(compute_metrics)(alg_name, k, n, ind) for k, n, ind in itertools.product(range(11), n_range, range(50)))

metrics_dict = dict(zip(itertools.product(range(11), n_range, range(50)), metrics_list))

with open('../res/%s/res-%s.pkl' %(alg_name, data_name), 'wb') as f:
    pickle.dump(results_dict, f)

with open('../res/%s/eval-%s.pkl' %(alg_name, data_name), 'wb') as f:
    pickle.dump(metrics_dict, f)