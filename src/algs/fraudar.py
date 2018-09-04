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
f, pathname, desc = imp.find_module('fraudar', sys.path[1:])
fraudar = imp.load_module('fraudar', f, pathname, desc)

data_name = sys.argv[1]
k = int(sys.argv[2])
N = int(sys.argv[3])
ind = int(sys.argv[4])

exec(open('fake_block.py', 'r').read())

outfile = '../res/fraudar/%s/%s-%d-%d-%d.csv' %(data_name, data_name, k, N, ind)

# the first graph is the type of fraud patterns
n_pattern = 8
graph = fraudar.ReviewGraph(n_pattern, fraudar.aveDegree)

reviewers = {n: graph.new_reviewer(n) for n in G.node if n.startswith('u')}
products = {n: graph.new_product(n) for n in G.node if n.startswith('p')}

for e in G.edges:
    graph.add_review(reviewers[e[0]], products[e[1]], (G.edges[e]['weight'] + 1)/2)

graph.update()

feagle_list = [[out_dict[r.name], r.name, r.anomalous_score] for r in graph.reviewers if r.name in out_dict]

out_list = [[x[0]] + ['s' + x[1][1:]] + x[2:] if x[1] in socks_list else x for x in feagle_list]
# fraud eagle output anomaly score instead of goodness score
out_list = sorted(out_list, key=lambda x: -x[2])
pd.DataFrame(out_list).to_csv(outfile, header=False, index=False)