#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import time, datetime

import fraud_eagle as feagle

data_name = sys.argv[1]
k = int(sys.argv[2])
N = int(sys.argv[3])
ind = int(sys.argv[4])

exec(open('fake_block.py', 'r').read())

outfile = '../res/feagle/%s/%s-%d-%d-%d.csv' %(data_name, data_name, k, N, ind)

epsilon = 0.25
graph = feagle.ReviewGraph(epsilon)

reviewers = {n: graph.new_reviewer(n) for n in G.node if n.startswith('u')}
products = {n: graph.new_product(n) for n in G.node if n.startswith('p')}

for e in G.edges:
    graph.add_review(reviewers[e[0]], products[e[1]], G.edges[e]['weight'])

for it in range(5):
    diff = graph.update()
    print('iter %d, diff %.2f' %(it, diff))
    if diff < 1e-3:
        print('early stop')
        break

feagle_list = [[out_dict[r.name], r.name, r.anomalous_score] for r in graph.reviewers if r.name in out_dict]

out_list = [[x[0]] + ['s' + x[1][1:]] + x[2:] if x[1] in socks_list else x for x in feagle_list]
# fraud eagle output anomaly score instead of goodness score
out_list = sorted(out_list, key=lambda x: -x[2])
pd.DataFrame(out_list).to_csv(outfile, header=False, index=False)