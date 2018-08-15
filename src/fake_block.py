#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import time, datetime
import pickle

def load_data(data_name):
    data_list = ['alpha', 'amazon', 'epinions', 'otc']
    assert data_name in data_list
    network_df = pd.read_csv('../rev2data/%s/%s_network.csv' %(data_name, data_name), header=None, names=['src', 'dest', 'rating', 'timestamp'], parse_dates=[3], infer_datetime_format=True)
    gt_df = pd.read_csv('../rev2data/%s/%s_gt.csv' %(data_name, data_name), header=None, names=['id', 'label'])
    if data_name in ['alpha', 'amazon', 'epinions', 'otc']:
        network_df['timestamp'] = pd.to_datetime(network_df['timestamp'], unit='s')
    return network_df, gt_df

network_df, gt_df = load_data(data_name)

print(network_df.shape)
print(gt_df.shape)
print('rating describe')
print(network_df['rating'].describe())

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

# target
np.random.seed(53)
T_index = np.random.permutation(len(prod_list))[ind]
T = prod_list[T_index]
# K sockpuppets, minimum is set to 1 if k is non-zero
K = int(np.ceil(k * count_dict[T] / 10))
# N geniune reviews for each sockpuppets
N = N

print('target product', T)
print('current avg rating', rating_dict[T])
print('num of rating', count_dict[T])

print('generate %d socks' %K)
print('%d reviews per sock' %N)

def generate_sockpuppets(base_index=0, num=1):
    # socks = np.arange(base_index, base_index+num).tolist()
    socks = []
    while len(socks) < num:
        r = np.random.randint(low=len(user_list), high=10*len(user_list))
        if r not in user_list: socks += [r]
    return socks

np.random.seed(79)
def generate_reviews(user, prod, prod_list, num):
    fr = rating_max
    if rating_dict[prod] > 0:
        fr = rating_min
    reviews = [[user, prod, fr, ts_max]]
    fr_prods = np.random.permutation(prod_list)[:num]
    reviews += [[user, p, np.clip(np.random.normal(rating_dict[p], std_dict[p], 1)[0], a_min=rating_min, a_max=rating_max), pd.datetime.today()] for p in fr_prods]
    return reviews

socks = generate_sockpuppets(len(user_list), K)

fake_data = []
for sock in socks:
    fake_reviews = generate_reviews(sock, T, prod_list, N)
    fake_data += fake_reviews

fake_df = pd.DataFrame(fake_data, columns=['src', 'dest', 'rating', 'timestamp'])
print('fake data shape', fake_df.shape)
fake_list = pd.DataFrame({'socks': socks, 'value':-1})

df = pd.concat([network_df, fake_df])
df['fairness'] = 1
df['src'] = 'u' + df['src'].astype(str)
df['dest'] = 'p' + df['dest'].astype(str)
df['weight'] = (df['rating'] - rating_min)/(rating_max - rating_min) * 2 - 1
df['timestamp'] = pd.to_numeric(df['timestamp']) / 1e9
print(df.shape)

new_rating_dict = {'p'+str(p): (rating_dict[p]-rating_min)/(rating_max-rating_min)*2-1 for p in rating_dict}

G = nx.from_pandas_edgelist(df, 'src', 'dest', ['weight', 'timestamp', 'fairness'], create_using=nx.DiGraph())
print('number of totdal nodes', len(G.nodes))
for node in G.nodes:
    if node.startswith('u'):
        G.node[node]['fairness'] = 1
    else:
        G.node[node]['goodness'] = new_rating_dict[node]

out_list = ['u'+str(u) for u in socks + gt_df['id'].tolist()]