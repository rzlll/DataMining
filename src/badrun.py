#!/usr/bin/env python3

import os, sys
import numpy as np
import pandas as pd

import sklearn
import networkx as nx

import time, datetime
import pickle

cmp = lambda x, y: (x > y) - (x < y)

data_name = sys.argv[1]
k = int(sys.argv[2])
N = int(sys.argv[3])
ind = int(sys.argv[4])

exec(open('fake_block.py', 'r').read())
outfile = '../badres/%s/%s-%d-%d-%d.csv' %(data_name, data_name, k, N, ind)

## algorithm begins here

iter_idx = 0
nodes = G.nodes()
edges = G.edges(data=True)
print("Num nodes = %d, edges = %d" %(len(nodes), len(edges)))

verbose = False

for n in nodes:
    if 'u' in n:
        G.node[n]['bias'] = 0.0
    else:
        G.node[n]['deserve'] = 0.0   

dr = 0
dt = 0
dh = 0

def bias():
    nodeList=G.nodes()
    for nodeID in nodeList:
        if "u" not in nodeID[0]:
            continue

        #Normalizing Factor
        norm=2*G.out_degree(nodeID)
        #print norm,nodeID

        #Sum of signs of outgoing link                                     
        outWeight=G.out_degree(nodeID,weight='weight')

        for suc in G.successors(nodeID):
            outWeight=outWeight-G.node[suc]['deserve']
        if norm==0:
            G.node[nodeID]['bias']=-2
        else:
            G.node[nodeID]['bias']=outWeight/norm

def deserve():
    nodeList=G.nodes()
    for node in nodeList:
        if "p" not in node[0]:
            continue

        #Normalizing Factor
        norm=G.in_degree(node)
        #Sum of signs of outgoing link
        inWeight=G.in_degree(node, weight='weight')

        for pre in G.predecessors(node):
            if G.node[pre]['bias']*G.edges[pre, node]['weight'] > 0:
                    inWeight = inWeight - cmp(G.edges[pre, node]['weight'],0)*G.node[pre]['bias'] * G.edges[pre, node]['weight']
        if norm==0:
            G.node[node]['deserve']=-2
        else:
            G.node[node]['deserve']=inWeight/norm

for iter in range(10):
    print("Iteration", iter)
    deserve()
    bias()
    iter += 1

goodness_vals = []
for node in nodes:
    if "u" not in node:
        continue
    f = 1 - abs(G.node[node]["bias"])
    goodness_vals.append([node, f, (0.5-f)*np.log(G.out_degree(node)+1), G.out_degree(node)])
goodness_vals = np.array(goodness_vals)
# sortedlist = sorted(goodness_vals, key= lambda x: (float(x[1]), -1*float(x[2])))

pd.DataFrame(goodness_vals).to_csv(outfile, header=False, index=False)

# fw = open("result/%s-bad-sorted-users.csv" % (data_name),"w")
# for gg in sortedlist:
#         fw.write("%s,%s,%s,%s\n" % (gg[-1], gg[0], gg[1], gg[2]))
# fw.close()
