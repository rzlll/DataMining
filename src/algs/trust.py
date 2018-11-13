#!/usr/bin/env python3

import os, sys
import math
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import time, datetime
import pickle


data_name = sys.argv[1]
k = int(sys.argv[2])
N = int(sys.argv[3])
ind = int(sys.argv[4])

exec(open('fake_block.py', 'r').read())
outfile = '../res/trust/%s/%s-%d-%d-%d.csv' %(data_name, data_name, k, N, ind)

## algorithm begins here

def normalise(num): # scale to the [-1, 1] range following logistic-like distribution
    try:
        return (2/(1+math.exp(-num)))-1
    except OverflowError:
        num = min(1, max(-1, -num))
        return num
        
        if num < -1:
            return -1
        else:
            return 1

def timediff(t1, t2):
    return abs(t1-t2)


iter_idx = 0
nodes = G.nodes()
edges = G.edges(data=True)
print("Num nodes = %d, edges = %d" %(len(nodes), len(edges)))

verbose = False

for n in nodes:
    if 'u' in n:
        G.node[n]['trustiness'] = 1
    else:
        G.node[n]['reliability'] = 1   

for e in edges:
    e[2]['honesty'] = 1
    
dr = 0
dt = 0
dh = 0
    
while iter_idx < 10:
    print("-----\niter: ", iter_idx)
    
    print(dr, dt, dh)
    
    dr = 0
    dt = 0
    dh = 0
    
    ################################################################
    cnt = 0
    # compute_honesty
    print('updating honesty of reviews')
    for edge in edges:
        # calc_agreement
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        t_agr = 0
        t_dis = 0
        iedges = G.in_edges(edge[1], data=True)
        for neigh_edge in iedges:
            # neigh_edge consists of other reviews for the same product
            if neigh_edge[0] == edge[0]:
                continue
        
            if data_name != "epinions" and timediff(neigh_edge[2]["timestamp"], edge[2]["timestamp"]) > 60*60*24*90:
                continue
            if abs(neigh_edge[2]['weight'] - edge[2]['weight']) <= 5/10.0:
                t_agr += G.node[neigh_edge[0]]['trustiness']
            else:
                t_dis += G.node[neigh_edge[0]]['trustiness']
        A = float(t_agr - t_dis)
        agreement = normalise(A)

        # calc_honesty
        h = abs(G.node[edge[1]]['reliability']) * agreement
        dh += abs(edge[2]['honesty'] - h)
        edge[2]['honesty'] = h
    
    ############################################################
    # compute_trustiness
    print("updating trustiness of users")
    for node in nodes:
        if 'u' not in node:
            continue
            
        # calc_trustiness
        H = sum([e[2]['honesty'] for e in G.out_edges(node, data=True)])
        T = normalise(float(H))
        if verbose: 
            print("Trustiness", node)
            print("H = ", H)
            print("T = ", T )
        dt += abs(G.node[node]['trustiness'] - T)
        G.node[node]['trustiness'] = T
    
    ############################################################
    # compute_reliability
    print('updating reliability of stores')
    for node in nodes:
        if "p" not in node:
            continue
            
        # calc_reliability
        theta = 0
        if verbose:
            print("Store_id", node)
        iedges =  G.in_edges(node, data=True)   
        for e in iedges:
            curr_trust = G.node[e[0]]['trustiness']
            if curr_trust > 0:
                theta += (e[2]['weight']*curr_trust)
                if verbose:
                    print("review rating = ", e[2]['weight'])
                    print("curr_trust = ", curr_trust)
                    print("theta = ", theta)
        R = normalise(theta)
        if verbose:
            print("Final Theta = ", theta)
            print("R = ", R)
        dr += abs(G.node[node]['reliability'] - R)
        G.node[node]['reliability'] = R
        
    if dr < 0.01 and dt < 0.01 and dh < 0.01:
        print('early stop at %d' %iter_idx)
        break
    
    iter_idx += 1

from operator import itemgetter
user_nodes = [n for n in G.nodes() if 'u' in n]
trustiness_scores = [G.node[n]['trustiness'] for n in user_nodes]
sortedlist_gw = sorted(zip(user_nodes, trustiness_scores), key=itemgetter(1))

trust_list = [[out_dict[x[0]]]+list(x) for x in sortedlist_gw if x[0] in out_dict]

out_list = [[x[0]] + ['s' + x[1][1:]] + x[2:] if x[1] in socks_list else x for x in trust_list]
out_list = sorted(out_list, key=lambda x: x[2])
pd.DataFrame(out_list).to_csv(outfile, header=False, index=False)

# fw = open("result/alpha-icdm2011-sorted-users2.csv","w")
# for gg in sortedlist_gw:
#   fw.write("%s,%f\n" % gg)
# fw.close()

# from operator import itemgetter
# user_nodes = [n for n in G.nodes() if 'p' in n]
# trustiness_scores = [G.node[n]['reliability'] for n in user_nodes]
# sortedlist_gw = sorted(zip(user_nodes, trustiness_scores), key=itemgetter(1))

# fw = open("result/alpha-icdm2011-sorted-products2.csv","w")
# for gg in sortedlist_gw:
#   fw.write("%s,%f\n" % gg)
# fw.close()
