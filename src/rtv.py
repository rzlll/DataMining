#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import time, datetime
import pickle


# ## parameters setting for rtv algorithm

# In[2]:


import argparse

parser = argparse.ArgumentParser(description='rtv algorithm')
# parser.add_argument('-d', '--diff', type=int, default=10, action='store', help='diff days')

if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    from tqdm import tqdm_notebook as tqdm
    data_name = 'alpha'
    alpha1 = 1
    alpha2 = 1
    beta1 = 1
    beta2 = 1
    gamma1 = 1
    gamma2 = 1
    gamma3 = 1
    gamma4 = 1
    max_iter = 10
    k = 1
    N = 1
    ind = 1
    trusted_num = 100
    verified_num = 100
    trusted_rev = 50
else:
    from tqdm import tqdm
    print('script mode')
    display=print
    
    data_name = sys.argv[1]

    alpha1 = int(sys.argv[2])
    alpha2 = int(sys.argv[3])

    beta1 = int(sys.argv[4])
    beta2 = int(sys.argv[5])

    gamma1 = int(sys.argv[6])
    gamma2 = int(sys.argv[7])
    gamma3 = int(sys.argv[8])
    gamma4 = int(sys.argv[9])

    max_iter = int(sys.argv[10])

    k = int(sys.argv[11])
    N = int(sys.argv[12])
    ind = int(sys.argv[13])
    
    trusted_num = 100
    verified_num = 100
    trusted_rev = 50


# In[3]:


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


# In[4]:


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


# In[5]:


gt_dict = dict(gt_df.values)
sd_list = network_df[['src', 'dest']].values.tolist()
target_pool_bad = set([t[1] for t in sd_list if t[0] in gt_dict and gt_dict[t[0]] == -1])
target_pool_good = set([t[1] for t in sd_list if t[0] in gt_dict and gt_dict[t[0]] == 1])
target_pool = list(target_pool_good & target_pool_bad)

print('Target pool size %d, index at %d' %(len(target_pool), ind))
if len(target_pool) <= ind:
    print('exit')


# In[6]:


np.random.seed(53)
T_index = np.random.permutation(len(target_pool))[ind]
T = target_pool[T_index]
# K sockpuppets, minimum is set to 1 if k is non-zero
K = int(np.ceil(k * count_dict[T] / 10))
# N geniune reviews for each sockpuppets
N = N

print('target product', T)
print('current avg rating', rating_dict[T])
print('num of rating', count_dict[T])


# In[7]:


print('generate %d socks' %K)
print('%d reviews per sock' %N)

def generate_accounts(base_index=0, num=1):
    # socks = np.arange(base_index, base_index+num).tolist()
    socks = []
    while len(socks) < num:
        r = np.random.randint(low=len(user_list), high=10*len(user_list))
        if r not in user_list: socks += [r]
    return socks

np.random.seed(79)
def generate_bad_reviews(user, prod, prod_list, num=1):
    assert num > 0
    fr = rating_max
    if rating_dict[prod] > 0:
        fr = rating_min
    reviews = [[user, prod, fr, ts_max]]
    fr_prods = np.random.permutation(prod_list)[:num-1]
    reviews += [[user, p, np.clip(np.random.normal(rating_dict[p], std_dict[p], 1)[0], a_min=rating_min, a_max=rating_max), pd.datetime.today()] for p in fr_prods]
    return reviews


# In[8]:


socks = generate_accounts(len(user_list), K)

fake_data = []
for sock in socks:
    fake_reviews = generate_bad_reviews(sock, T, prod_list, N)
    fake_data += fake_reviews

fake_df = pd.DataFrame(fake_data, columns=['src', 'dest', 'rating', 'timestamp'])
print('fake data shape', fake_df.shape)
# fake_list = pd.DataFrame({'socks': socks, 'value':-1})


# In[9]:


np.random.seed(89)
trusted_users = []
while len(trusted_users) < trusted_num:
    r = np.random.randint(low=len(user_list), high=10*len(user_list))
    if r not in user_list + socks: trusted_users += [r]

def generate_good_reviews(user, prod, prod_list, num=1):
    assert num > 0
    fr_prods = np.random.permutation(prod_list)[:num]
    reviews = [[user, p, np.clip(np.random.normal(rating_dict[p], std_dict[p]*0.2, 1)[0], a_min=rating_min, a_max=rating_max), pd.datetime.today()] for p in fr_prods]
    return reviews

trusted_data = []
for tuser in trusted_users:
    trusted_data += generate_good_reviews(tuser, T, prod_list, trusted_rev)

trusted_df = pd.DataFrame(trusted_data, columns=['src', 'dest', 'rating', 'timestamp'])


# In[10]:


np.random.seed(83)

verified_index = np.random.permutation(len(user_list))[:verified_num]
verified_users = [user_list[v] for v in verified_index]


# In[11]:


df = pd.concat([network_df, fake_df, trusted_df])
df['fairness'] = 1
df['src'] = 'u' + df['src'].astype(str)
df['dest'] = 'p' + df['dest'].astype(str)
df['weight'] = (df['rating'] - rating_min)/(rating_max - rating_min) * 2 - 1
df['timestamp'] = pd.to_numeric(df['timestamp']) / 1e9
print(df.shape)

new_rating_dict = {'p'+str(p): (rating_dict[p]-rating_min)/(rating_max-rating_min)*2-1 for p in rating_dict}


# In[12]:


G = nx.from_pandas_edgelist(df, 'src', 'dest', ['weight', 'timestamp', 'fairness'], create_using=nx.DiGraph())
print('number of totdal nodes', len(G.nodes))
for node in G.nodes:
    if node.startswith('u'):
        G.node[node]['fairness'] = 1
    else:
        G.node[node]['goodness'] = new_rating_dict[node]


# In[13]:


out_dict = {'u'+str(t[0]): gt_dict[t[0]] for t in sd_list if t[1] == T and t[0] in gt_dict}
out_dict.update({'u'+str(u): -1 for u in socks})

socks_list = ['u'+str(u) for u in socks]
trusted_list = ['u' + str(u) for u in trusted_users]
verified_list = ['u' + str(u) for u in verified_users]


# In[14]:


outdir = '../res/rtv/%s' %(data_name)
print('save to', outdir)

## algorithm begins here

nodes = G.nodes()
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


# In[15]:


for node in nodes:
    if node[0] != 'u': continue
    G.nodes[node]['mu'] = np.mean([e[2]['weight'] for e in G.out_edges(node, data=True)])


# In[16]:


for edge in edges:
    dsc = G.edges[edge[0], edge[1]]['weight'] - G.node[edge[0]]['mu']
    if dsc == 0:
        G.edges[edge[0], edge[1]]["sc"] = 0
    elif dsc > 0:
        G.edges[edge[0], edge[1]]["sc"] = np.clip(dsc / (1-G.node[edge[0]]['mu']), -1, 1)
    else:
        G.edges[edge[0], edge[1]]["sc"] = np.clip(dsc / (1+G.node[edge[0]]['mu']), -1, 1)


# In[17]:


te = [e for e in edges if e[0] in trusted_list]
print(len(te))
print(te[:10])

ve = [e for e in edges if e[0] in verified_list]
print(len(ve))
print(ve[:10])


# In[18]:


# print(edge)
# print([e[0] in trusted_list for e in G.in_edges(edge[1])])
# print([1 - abs(edge[2]['sc'] - G.edges[e]['sc'])/2.0 for e in G.in_edges(edge[1]) if e != edge[:2] if e[0] in trusted_list])
# print([e != edge[:2] for e in G.in_edges(edge[1])])


# In[27]:


##### RTV ITERATIONS START ######
iter = 0
du=0
dp=0
dr=0

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
        # if the user is truster, the reliability of the rating is zaro
        if G.node[edge[0]] in trusted_list:
            G.edges[edge[0], edge[1]]["fairness"] = 1
            continue
        
        component1 = []
        component2 = []
        component3 = []
        for e in G.in_edges(edge[1]):
            if e == edge[:2]:
                continue
            v = 1 - abs(edge[2]['sc'] - G.edges[e]['sc'])/2.0
            v = v * G.node[edge[0]]['fairness']
            if e[0] in trusted_list:
                component1 += [v]
            elif e[0] in verified_list:
                component2 += [v]
            else:
                component3 += [v]

        # rating_distance = 1 - (abs(edge[2]["weight"] - G.node[edge[1]]["goodness"])/2.0)
        # user_fairness = G.node[edge[0]]["fairness"]
        # ee = (edge[0], edge[1])
        kl_text = 1.0 - full_birdnest_edge[edge_map[e[:2]]]

        # the equation for REV2
        # x = (gamma2*rating_distance + gamma1*user_fairness + gamma3*kl_text)/(gamma1 + gamma2 + gamma3)
        # the equation for RTV
        c1 = gamma1*np.mean(component1) if len(component1) > 0 else 0
        c2 = gamma2*np.mean(component2) if len(component2) > 0 else 0
        c3 = gamma3*np.mean(component3) if len(component3) > 0 else 0
        
        x = (c1 + c2 + c3 + gamma4*kl_text)/(gamma1*len(component1) + gamma2*len(component2) + gamma3*len(component3) + gamma4)
        
        if x < 0.00:
            x = 0.0
        if x > 1.0:
            x = 1.0

        dr += abs(edge[2]["fairness"] - x)
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
        # the trusted users have fairness = 1
        if node in trusted_list:
            G.node[node]["fairness"] = 1.0
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


# In[37]:


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
# all_node_vals = np.array(all_node_vals)

outfile = '%s/%s-%d-%d-%d-%d-%d-%d-%d-%d-%d-%d.csv' % (outdir, data_name, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, k, N, ind)

rtv_list = [[out_dict[x[0]]]+ list(x) for x in all_node_vals if x[0] in out_dict]

out_list = [[x[0]] + ['s' + x[1][1:]] + x[2:] if x[1] in socks_list else x for x in rtv_list]

out_list = [[x[0]] + ['t' + x[1][1:]] + x[2:] if x[1] in trusted_list else x for x in out_list]
out_list = [[x[0]] + ['v' + x[1][1:]] + x[2:] if x[1] in verified_list else x for x in out_list]

out_list = sorted(out_list, key=lambda x: x[2])


# In[38]:


print(out_list)


# In[40]:


print(pd.DataFrame(out_list))


# In[39]:


pd.DataFrame(out_list).to_csv(outfile, header=False, index=False)

