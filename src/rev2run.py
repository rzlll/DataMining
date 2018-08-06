#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import time, datetime
import pickle
from collections import defaultdict

data_name = sys.argv[1]

alpha1 = int(sys.argv[2])
alpha2 = int(sys.argv[3])

beta1 = int(sys.argv[4])
beta2 = int(sys.argv[5])

gamma1 = int(sys.argv[6])
gamma2 = int(sys.argv[7])
gamma3 = int(sys.argv[8])

max_iter = int(sys.argv[9])

k = int(sys.argv[10])
N = int(sys.argv[11])
ind = int(sys.argv[12])

if gamma1 == 0 and gamma2 == 0 and gamma3 == 0: sys.exit(0)

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
# print('reviews/prod %.2f' %rev_per_prod)
# print('min max %.2f %.2f' %(rating_min, rating_max))

# target
np.random.seed(53)
T_index = np.random.permutation(len(prod_list))[ind]
T = network_df['dest'][T_index]
# K sockpuppets
K = int(k * count_dict[T] / 10)
# N geniune reviews for each sockpuppets
N = N

print('target product', T)
print('current avg rating', rating_dict[T])
print('num of rating', count_dict[T])

print('generate %d socks' %K)
print('%d reviews per sock' %N)

def generate_sockpuppets(base_index=0, num=1):
    socks = np.arange(base_index, base_index+num).tolist()
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
print(df.shape)

new_rating_dict = {'p'+str(p): (rating_dict[p]-rating_min)/(rating_max-rating_min)*2-1 for p in rating_dict}

G = nx.from_pandas_edgelist(df, 'src', 'dest', ['weight', 'timestamp', 'fairness'], create_using=nx.DiGraph())
print('number of totdal nodes', len(G.nodes))
for node in G.nodes:
    if node.startswith('u'):
        G.node[node]['fairness'] = 1
    else:
        G.node[node]['goodness'] = new_rating_dict[node]

outdir = '../rev2res/%s' %(data_name)
print('save to', outdir)

## algorithm begins here

nodes = G.nodes()
# nodes = list(G.nodes)  # To solve AttributeError
edges = G.edges(data=True)
print ("%s network has %d nodes and %d edges" % (data_name, len(nodes), len(edges)))

user_names = [node for node in nodes if "u" in node]
product_names = [node for node in nodes if "p" in node]
num_users = len(user_names)
num_products = len(product_names)
user_map = dict(zip(user_names, range(len(user_names))))
product_map = dict(zip(product_names, range(len(product_names))))

full_birdnest_user = [0] * len(user_names)
full_birdnest_product = [0] * len(product_names)
full_birdnest_edge = []
print ("Init birdnest for %s" % data_name)
full_birdnest_edge = [0.0]*len(edges)

# adapted to nx v2
edges_arr = nx.convert_matrix.to_pandas_edgelist(G).values
ae = zip(edges_arr[:, 0], edges_arr[:, 1])
edge_map = dict(zip(ae, range(len(edges))))

for node in nodes:
    if "u" in node[0]:
        G.node[node]["fairness"] = 1 - full_birdnest_user[user_map[node]]
    else:
        G.node[node]["goodness"] = (1 - full_birdnest_product[product_map[node]] - 0.5)*2

for edge in edges:
    G[edge[0]][edge[1]]["fairness"] = 1 - full_birdnest_edge[edge_map[(edge[0], edge[1])]]

yfg = []
ygood = []
xfg = []
du = 0
dp = 0
dr = 0

##### REV2 ITERATIONS START ######
iter = 0
while iter < max_iter:
    print ('-----------------')
    print ("Epoch number %d with du = %f, dp = %f, dr = %f, for (%d,%d,%d,%d,%d,%d,%d)" % (iter, du, dp, dr, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3))
    if np.isnan(du) or np.isnan(dp) or np.isnan(dr): break
    
    du = 0
    dp = 0
    dr = 0
    
    ############################################################

    print ('Update goodness of product')

    currentgvals = []
    for node in nodes:
        if "p" not in node[0]:
            continue
        currentgvals.append(G.node[node]["goodness"])
    
    median_gvals = np.median(currentgvals) # Alternatively, we can use mean here, intead of median

    for node in nodes:
        if "p" not in node[0]:
            continue
        
        inedges = G.in_edges(node,  data=True)
        ftotal = 0.0
        gtotal = 0.0
        for edge in inedges:
            gtotal += edge[2]["fairness"]*edge[2]["weight"]
        ftotal += 1.0
        
        kl_timestamp = ((1 - full_birdnest_product[product_map[node]]) - 0.5)*2

        if ftotal > 0.0:
            mean_rating_fairness = (beta1*median_gvals + beta2* kl_timestamp + gtotal)/(beta1 + beta2 + ftotal)
        else:
            mean_rating_fairness = 0.0
        
        x = mean_rating_fairness
        
        if x < -1.0:
            x = -1.0
        if x > 1.0:
            x = 1.0
        dp += abs(G.node[node]["goodness"] - x)
        G.node[node]["goodness"] = x
    
    ############################################################
    
    print ("Update fairness of ratings")
    for edge in edges:
        rating_distance = 1 - (abs(edge[2]["weight"] - G.node[edge[1]]["goodness"])/2.0)
        
        user_fairness = G.node[edge[0]]["fairness"]
        ee = (edge[0], edge[1])
        kl_text = 1.0 - full_birdnest_edge[edge_map[ee]]

        x = (gamma2*rating_distance + gamma1*user_fairness + gamma3*kl_text)/(gamma1+gamma2 + gamma3)

        if x < 0.00:
            x = 0.0
        if x > 1.0:
            x = 1.0
        
        dr += abs(edge[2]["fairness"] - x)
        # adapt to nx v2
        G.edges[edge[0], edge[1]]["fairness"] = x
    
    ############################################################
    
    currentfvals = []
    for node in nodes:
        if "u" not in node[0]:
            continue
        currentfvals.append(G.node[node]["fairness"])
        median_fvals = np.median(currentfvals) # Alternatively, we can use mean here, intead of median

    print ('update fairness of users')
    for node in nodes:
        if "u" not in node[0]:
            continue
        
        outedges = G.out_edges(node, data=True)
        
        f = 0
        rating_fairness = []
        for edge in outedges:
            rating_fairness.append(edge[2]["fairness"])
        
        for x in range(0,alpha1):
            rating_fairness.append(median_fvals)

        kl_timestamp = 1.0 - full_birdnest_user[user_map[node]]

        for x in range(0, alpha2):
            rating_fairness.append(kl_timestamp)

        mean_rating_fairness = np.mean(rating_fairness)

        x = mean_rating_fairness #*(kl_timestamp)
        if x < 0.00:
            x = 0.0
        if x > 1.0:
            x = 1.0

        du += abs(G.node[node]["fairness"] - x)
        G.node[node]["fairness"] = x
        #print mean_rating_fairness, kl_timestamp
    
    iter += 1
    if du < 0.01 and dp < 0.01 and dr < 0.01:
        break

### SAVE THE RESULT

currentfvals = []
for node in nodes:
    if "u" not in node[0]: # only store scores for edge generating nodes
        continue
    currentfvals.append(G.node[node]["fairness"])
median_fvals = np.median(currentfvals)
print(len(currentfvals), median_fvals)

all_node_vals = []
for node in nodes:
    if "u" not in node[0]:
        continue
    f = G.node[node]["fairness"]
    all_node_vals.append([node, (f - median_fvals)*np.log(G.out_degree(node)+1), f, G.out_degree(node)])
all_node_vals = np.array(all_node_vals)

outfile = '%s/%s-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d.csv' % (outdir, data_name, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, k, N, ind)

pd.DataFrame(all_node_vals).to_csv(outfile, header=False, index=False)
print('saved to %s' %outfile)
