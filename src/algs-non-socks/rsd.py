#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import time, datetime

# the normal import doesn't work, because this script has exactly same name as the module
# import fraudar
import imp
f, pathname, desc = imp.find_module('rsd', sys.path[1:])
rsd = imp.load_module('rsd', f, pathname, desc)

data_name = sys.argv[1]
k = 0
N = 0
ind = 0

exec(open('fake-non-socks.py', 'r').read())
outfile = '../res/non-socks/rsd-%s.csv' %data_name

theta = 0.25
graph = rsd.ReviewGraph(theta)

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

rsd_list = [[out_dict[r.name], r.name, r.anomalous_score] for r in graph.reviewers if r.name in out_dict]

out_list = [[x[0]] + ['s' + x[1][1:]] + x[2:] if x[1] in socks_list else x for x in rsd_list]
# fraud eagle output anomaly score instead of goodness score
out_list = sorted(out_list, key=lambda x: -x[2])
pd.DataFrame(out_list).to_csv(outfile, header=False, index=False)