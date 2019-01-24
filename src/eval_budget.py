#!/usr/bin/env python3

import os, sys, argparse
import numpy as np
import pandas as pd

import numba
import sklearn
import sklearn.metrics

# import time, datetime
import itertools
# from joblib import Parallel, delayed
import pickle

parser = argparse.ArgumentParser(description='evaluate algorithms with data')
parser.add_argument('-d', '--data', type=str, default='alpha', choices=['alpha', 'amazon', 'epinions', 'otc'],
                    help='data name')
parser.add_argument('-a', '--alg', type=str, default='bad',
                    choices=['bn', 'feagle', 'fraudar', 'trust', 'rsd', 'bad', 'rev2'], help='alg name')
parser.add_argument('-n', '--ncores', type=int, default=1, help='number of cores to use (disabled after using numba)')

# for budget model
parser.add_argument('-c', '--cost', type=int, default=1, help='cost')
parser.add_argument('-b', '--budget', type=int, default=1, help='total budget')

parsed = parser.parse_args(sys.argv[1:])

if not os.path.exists('../res/%s/%s' % (parsed.alg, parsed.data)):
    print('no results for %s %s' % (parsed.alg, parsed.data))
    exit(-1)

data_name = parsed.data
alg_name = parsed.alg
n_cores = parsed.ncores

# ccost_list = [6, 7, 8, 10]
# budget_list = [100, 300, 500]

ccost_budget_pairs = [(6, 600), (6.5, 325), (7, 70), (8, 40), (6, 1200), (6, 1000)]

print(alg_name, data_name)
print(ccost, budget)

def load_data(data_name):
    data_list = ['alpha', 'amazon', 'epinions', 'otc']
    assert data_name in data_list
    network_df = pd.read_csv('../rev2data/%s/%s_network.csv' %(data_name, data_name), header=None, names=['src', 'dest', 'rating', 'timestamp'], parse_dates=[3], infer_datetime_format=True)
    gt_df = pd.read_csv('../rev2data/%s/%s_gt.csv' %(data_name, data_name), header=None, names=['id', 'label'])
    if data_name in ['alpha', 'amazon', 'epinions', 'otc']:
        network_df['timestamp'] = pd.to_datetime(network_df['timestamp'], unit='s')
    return network_df, gt_df

network_df, gt_df = load_data(data_name)

ts_max = network_df['timestamp'].max()
user_list = network_df['src'].unique().tolist()
prod_list = network_df['dest'].unique().tolist()
rev_per_prod = network_df.shape[0]/len(prod_list)
rating_dict = network_df.groupby('dest')['rating'].mean().to_dict()
count_dict = network_df.groupby('dest')['rating'].count().to_dict()
std_dict = network_df.groupby('dest')['rating'].std().fillna(0).to_dict()

rating_max = network_df['rating'].max()
rating_min = network_df['rating'].min()

print('users %d' %len(user_list))
print('products %d' %len(prod_list))
print('reviews %d' %network_df.shape[0])

# get the target pool
gt_dict = dict(gt_df.values)
sd_list = network_df[['src', 'dest']].values.tolist()
target_pool_bad = set([t[1] for t in sd_list if t[0] in gt_dict and gt_dict[t[0]] == -1])
target_pool_good = set([t[1] for t in sd_list if t[0] in gt_dict and gt_dict[t[0]] == 1])
target_pool = list(target_pool_good & target_pool_bad)

n_range = list(range(0, 51, 5))
n_range[0] = 1

d_list = list(range(50))
np.random.seed(53)
T_index = np.random.permutation(len(target_pool))[d_list]

# compute the bugets pairs

ccost_budegt_eligibles = {}

for ccost, budget in ccost_budget_pairs:
    eligible_bugets = []
    for k, n in itertools.product(range(10), n_range):
        for i in d_list:
            t = target_pool[i]
            K = int(np.ceil(k * count_dict[t] / 10))
            sum_cost = K * ccost * n
            if sum_cost <= budget:
                eligible_bugets += [(k, n, i)]
    cost_budget_eligibles[(ccost, budget)] = eligible_bugets

def compute_score(k=0, n=0, ind=0):
    # if user is good in ground truth output 0
    # if user is fraudster in ground truth output 1
    # if user is sockpuppet output 2

    if alg_name == 'rev2':
        results_df = pd.read_csv(
            '../res/%s/%s/%s-1-1-1-1-1-1-0-%d-%d-%d.csv' % (alg_name, data_name, data_name, k, n, ind), header=None)
        ulist = results_df[1].tolist()
        ytrue = [0 if t == 1 else 1 for t in results_df[0].tolist()]
        u_sum = {u: 0 for u in ulist}
        flist = [
            '../res/%s/%s/%s-1-1-1-1-1-1-0-%d-%d-%d.csv' % (alg_name, data_name, data_name, k, n, ind),
            '../res/%s/%s/%s-1-2-1-1-1-1-0-%d-%d-%d.csv' % (alg_name, data_name, data_name, k, n, ind),
            '../res/%s/%s/%s-1-1-2-1-1-1-0-%d-%d-%d.csv' % (alg_name, data_name, data_name, k, n, ind),
            '../res/%s/%s/%s-1-1-1-2-1-1-0-%d-%d-%d.csv' % (alg_name, data_name, data_name, k, n, ind),
            '../res/%s/%s/%s-1-1-1-1-2-1-0-%d-%d-%d.csv' % (alg_name, data_name, data_name, k, n, ind)
        ]
        for f in flist:
            try:
                try_df = pd.read_csv(f, header=None)
                s = dict(zip(try_df[1].tolist(), try_df[2].tolist()))
                for u in u_sum:
                    u_sum[u] += s[u]
            except:
                pass
        yscore = [u_sum[u] for u in u_sum]
    else:
        try:
            results_df = pd.read_csv('../res/%s/%s/%s-%d-%d-%d.csv' % (alg_name, data_name, data_name, k, n, ind),
                                     header=None)
        except:
            return None
        ytrue = [0 if t == 1 else 1 for t in results_df[0].tolist()]
        ulist = results_df[1].tolist()
        yscore = results_df[2].tolist()
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
        cut = qq * size
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

print('retrieve results and compute the metrics')

budget_results = {}
budget_metrics = {}

for ccost, budget in ccost_budget_pairs:
    print('cost: {}, budget: {}'.format(ccost, budget))
    
    results_dict = {}
    metrics_dict = {}
    
    eligible_bugets = cost_budget_eligibles[(ccost, budget)]
    eligible_kn = set()
    for k, n, d in itertools.product(range(10), n_range, range(50)):
        if (k, n, d) in eligible_bugets:
            results_dict[(k, n, d)] = compute_score(k, n, d)
        else:
            results_dict[(k, n, d)] = None

        if results_dict[(k, n, d)] != None:
            metrics_dict[(k, n, d)] = compute_metrics(results_dict, k, n, d)
            if k > 0:
                eligible_kn.add((k, n))
        else:
            metrics_dict[(k, n, d)] = None

    print('eligible kn {}'.format(len(eligible_kn)))
    budget_results[(ccost, budget)] = results_dict
    budget_metrics[(ccost, budget)] = metrics_dict

print('save the pickles')
with open('../res/%s/budget-res-%s.pkl' % (alg_name, data_name), 'wb') as f:
    pickle.dump(budget_results, f)

with open('../res/%s/budget-eval-%s.pkl' % (alg_name, data_name), 'wb') as f:
    pickle.dump(budget_metrics, f)
