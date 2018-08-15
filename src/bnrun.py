#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import time, datetime
import pickle
import birdnestlib as bn

data_name = sys.argv[1]
k = int(sys.argv[2])
N = int(sys.argv[3])
ind = int(sys.argv[4])

exec(open('fake_block.py', 'r').read())

outfile = '../bnres/%s/%s-%d-%d-%d.csv' %(data_name, data_name, k, N, ind)

## algorithm begins here

num_ratings = 21
num_iat_bins = 49

def get_timestamp_dist(node, normalize):#modified (added 2nd argument)
    if 'u' in node[0]:
        edges = G.out_edges(node, data=True)
        ts = [edge[2]['timestamp'] for edge in edges]
        ts = np.array(sorted(ts))
    else:
        edges = G.in_edges(node, data=True)
        ts = [edge[2]['timestamp'] for edge in edges]
        ts = np.array(sorted(ts))
    
    diff_ts = ts[1:] - ts[:-1]
    
    y, x =  np.histogram(diff_ts, bins=np.logspace(0, 7, num_iat_bins+1))
    if sum(y)!=0.0 and normalize:
        y = y*1.0/sum(y)
    return x,y

def get_rating_dist(node, normalize):
    if 'u' in node[0]:
        edges = G.out_edges(node, data=True)
        ts = [int(round(10*edge[2]['weight'])) for edge in edges]
    else:
        edges = G.in_edges(node, data=True)
        ts = [int(round(10*edge[2]['weight'])) for edge in edges]
    
    y, x =  np.histogram(ts, bins=range(-10, 12))
    if sum(y)!=0.0 and normalize:
        y = y*1.0/sum(y)
    return x, y

nodes = G.nodes()
timestamp_distributions = {}

#modified (multiple changes in this block)
user_names = [node for node in nodes if 'u' in node]
product_names = [node for node in nodes if 'p' in node]

num_users = len(user_names)
num_products = len(product_names)

# maps for getting the index of each user/product based on its name
user_map = dict(zip(user_names, range(len(user_names))))
product_map = dict(zip(product_names, range(len(product_names))))

rev_user_map = dict(zip(range(len(user_names)), user_names))
rev_product_map = dict(zip(range(len(product_names)), product_names))

full_birdnest_user = []
full_birdnest_product = []

if True:
    user_timestamp_mat = np.zeros((num_users, num_iat_bins))
    product_timestamp_mat = np.zeros((num_products, num_iat_bins))

    print('Get timestamp distribution')
    for node in nodes:
        if sys.argv[1] == 'epinions':
            break
        x, y = get_timestamp_dist(node, False)
        if 'u' in node[0]:
            user_timestamp_mat[user_map[node], :] = y
        else:
            product_timestamp_mat[product_map[node], :] = y
        
        if sum(y)!=0.0:
            y = y*1.0/sum(y)
        timestamp_distributions[node] = y
    
    user_rating_mat = np.zeros((num_users, num_ratings))
    product_rating_mat = np.zeros((num_products, num_ratings))

    print('Get rating distribution')
    for node in nodes:
        x, y = get_rating_dist(node, False)
        if 'u' in node[0]:
                user_rating_mat[user_map[node], :] = y
        else:
                product_rating_mat[product_map[node], :] = y
        if sum(y)!=0.0:
                y = y*1.0/sum(y)
        
    if sys.argv[1] == 'epinions':
        full_birdnest_user = zip(range(num_users), bn.detect(user_rating_mat, user_timestamp_mat, False, 1))
        # full_birdnest_product = zip(range(num_users), bn.detect(product_rating_mat, product_timestamp_mat, True, False, 1))
    else:
        full_birdnest_user = zip(range(num_users), bn.detect(user_rating_mat, user_timestamp_mat, True, 2))
        # full_birdnest_product = zip(range(num_users), bn.detect(product_rating_mat, product_timestamp_mat, True, True, 1))

full_birdnest_user_scores = [(rev_user_map[ent[0]], ent[1]) for ent in full_birdnest_user]
bn_list = [x for x in full_birdnest_user_scores if x[0] in out_list]
pd.DataFrame(bn_list).to_csv(outfile, header=False, index=False)

# sorted_full_birdnest_user = sorted(full_birdnest_user, key=lambda x: x[1])
# sorted_full_birdnest_product = sorted(full_birdnest_product, key=lambda x: x[1])
# print sorted_full_birdnest_user[:10], sorted_full_birdnest_user[-10:]
# fw = open('result/%s-birdnest-sorted-products.csv' % (data_name),'w')
# for ent in sorted_full_birdnest_product:
#     usr = rev_product_map[ent[0]]
#         fw.write('%s,%f,%d\n' %(usr, ent[1], G.in_degree(usr)))
# fw.close()

# fw = open('result/%s-birdnest-sorted-users.csv' % (data_name),'w')
# for ent in sorted_full_birdnest_user:
#     usr = rev_user_map[ent[0]]
#         fw.write('%s,%f,%d\n' %(usr, ent[1], G.out_degree(usr)))
# fw.close()
