#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import pandas as pd

import numba
import sklearn
import sklearn.metrics

import time, datetime
import itertools
from joblib import Parallel, delayed
import pickle

parser = argparse.ArgumentParser(description='evaluate algorithms with data')
parser.add_argument('-d', '--data', type=str, default='alpha', choices=['alpha', 'amazon', 'epinions', 'otc'], help='data name')
parser.add_argument('-a', '--alg', type=str, default='bad', choices=['bn', 'feagle', 'fraudar', 'trust', 'rsd', 'bad', 'rev2'], help='alg name')
parser.add_argument('-n', '--ncores', type=int, default=1, help='number of cores to use (disabled after using numba)')


###Pick first [-t] target for comparison of Rev2, RTV and RTV-supervised 
parser.add_argument('-t', '--ntarget', type=int, default=5, help='number of targets to select')


parsed = parser.parse_args(sys.argv[1:])

if not os.path.exists('../res/%s/%s' %(parsed.alg, parsed.data)):
    print('no results for %s %s' %(parsed.alg, parsed.data))
    exit(-1)

data_name = parsed.data
alg_name = parsed.alg
n_cores = parsed.ncores
n_target = parsed.ntarget

print(alg_name, data_name)

network_df = pd.read_csv('../rev2data/%s/%s_network.csv' %(data_name, data_name), header=None, names=['src', 'dest', 'rating', 'timestamp'], parse_dates=[3], infer_datetime_format=True)
user_list = ['u' + str(u) for u in network_df['src'].tolist()]
gt_df = pd.read_csv('../rev2data/%s/%s_gt.csv' %(data_name, data_name), header=None, names=['id', 'label'])
gt = dict([('u'+str(x[0]), x[1]) for x in zip(gt_df['id'], gt_df['label'])])

def average_multiple(flist):
    results_df = pd.read_csv(flist[0], header=None)
    ulist = results_df[1].tolist()
    ytrue_old = results_df[0].tolist()
    ytrue = [0 if ytrue_old[i] == 1 else 2 if ulist[i][0] == 's' else 1 for i in range(len(ytrue_old))]
    u_sum = {u: 0 for u in ulist}
    for f in flist:
        try:
            try_df = pd.read_csv(f, header=None)
            s = dict(zip(try_df[1].tolist(), try_df[2].tolist()))
            for u in u_sum:
                u_sum[u] += s[u]
        except:
            pass
    yscore = [u_sum[u] for u in u_sum]
    return ulist, ytrue, yscore

def resort(ulist, ytrue, yscore):
    uscore = dict(zip(ulist, yscore))
    utrue = dict(zip(ulist, ytrue))
    slist = sorted(uscore, key=lambda u: uscore[u])
    strue = [utrue[u] for u in slist]
    sscore = [uscore[u] for u in slist]
    return slist, strue, sscore

def compute_score(k=0, n=0, ind=0):
    # if user is good in ground truth output 0
    # if user is fraudster in ground truth output 1
    # if user is sockpuppet output 2
    
    if alg_name == 'rev2':
        flist = [
            '../res/%s/%s/%s-1-1-1-1-1-1-0-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind),
            '../res/%s/%s/%s-1-2-1-1-1-1-0-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind),
            '../res/%s/%s/%s-1-1-2-1-1-1-0-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind),
            '../res/%s/%s/%s-1-1-1-2-1-1-0-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind),
            '../res/%s/%s/%s-1-1-1-1-2-1-0-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind)
        ]
        try:
            results_df = pd.read_csv(flist[0], header=None)
        except:
            return None
        
        ulist, ytrue, yscore = average_multiple(flist)
    else:
        try:
            results_df = pd.read_csv('../res/%s/%s/%s-%d-%d-%d.csv' %(alg_name, data_name, data_name, k, n, ind), header=None)
        except:
            return None
<<<<<<< HEAD
        ytrue = [0 if t==1 else 1 for t in results_df[0].tolist()]       
=======
        ytrue_old = results_df[0].tolist()
>>>>>>> ff3dc3ee56b8108b5deb5022c9fd6f535ddb2f5f
        ulist = results_df[1].tolist()
        yscore = results_df[2].tolist()
        ytrue = [0 if ytrue_old[i] == 1 else 2 if ulist[i][0] == 's' else 1 for i in range(len(ytrue_old))]
    
    ulist, ytrue, yscore = resort(ulist, ytrue, yscore)
    return {'ulist': ulist, 'ytrue': ytrue, 'yscore': yscore}




@numba.jit
def get_metrics(ytrue, yscore):
    '''get precision and recall at q percentile'''
    q = np.array([0.005, 0.01, 0.03, 0.05, 0.1])

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
    return prec_dict, recl_dict, f1_dict, prec1_dict

def compute_metrics(res_dict, k, n, ind):
    ulist = np.array(res_dict[(k, n, ind)]['ulist'])
    yscore = np.array(res_dict[(k, n, ind)]['yscore'])
    ytrue = np.array(res_dict[(k, n, ind)]['ytrue'])
    ytrue[ytrue > 1] = 1
    prec_dict, recl_dict, f1_dict = get_metrics(ytrue, yscore)
    return {'prec': prec_dict, 'recl': recl_dict, 'f1': f1_dict}

n_range = list(range(0, 51, 5))
n_range[0] = 1

results_dict = {}
metrics_dict = {}

print('retrieve results and compute the metrics')
for k, n, d in itertools.product(range(10), n_range, range(n_target)):  #range(50)
    results_dict[(k, n, d)] = compute_score(k, n, d)
    if results_dict[(k, n, d)] != None:
        metrics_dict[(k, n, d)] = compute_metrics(results_dict, k, n, d)
    else:
        metrics_dict[(k, n, d)] = None

print('save the pickles')
with open('../res/%s/res-%s.pkl' %(alg_name, data_name), 'wb') as f:
    pickle.dump(results_dict, f)

with open('../res/%s/eval-%s.pkl' %(alg_name, data_name), 'wb') as f:
    pickle.dump(metrics_dict, f)
