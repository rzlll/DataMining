#
# Code for running the REV2 algorithm from the paper "REV2: Fraudulent User Prediction in Rating Platforms"
#

import os
import sys

NETWORKNAME = sys.argv[1]

alpha1 = int(sys.argv[2])
alpha2 = int(sys.argv[3])

beta1 = int(sys.argv[4])
beta2 = int(sys.argv[5])

gamma1 = int(sys.argv[6])
gamma2 = int(sys.argv[7])
gamma3 = int(sys.argv[8])

if gamma1 == 0 and gamma2 == 0 and gamma3 == 0:
		sys.exit(0)

import random
import subprocess 
import numpy
import time
from collections import defaultdict
import csv
import networkx as nx
import datetime
import math
from math import exp
import unicodecsv 
import csv
#import detect
#import cpickle
import pickle
#import detect_iat

print ("Loading %s network" % NETWORKNAME)
G = pickle.load(open("../rev2data/%s/%s_network.pkl" % (NETWORKNAME, NETWORKNAME), "rb"))
with open("../rev2data/%s/%s_networkv2.pkl" % (NETWORKNAME, NETWORKNAME), 'wb') as file:
	pickle.dump([G.nodes(data=True), G.edges(data=True)], file)
print ('migrate nx v1 to v2', "../rev2data/%s/%s_network.pkl" % (NETWORKNAME, NETWORKNAME))
print ('exit')
exit()

print ("Loaded")

if NETWORKNAME in ["otc", "alpha"]:
	num_ratings = 21
else:
	num_ratings = 5
num_iat_bins = 49

def get_timestamp_dist(node, normalize):#modified (added 2nd argument)
	if "u" in node[0]:
		edges = G.out_edges(node, data=True)
		ts = [edge[2]["timestamp"] for edge in edges]
		ts = numpy.array(sorted(ts))
	else:
		edges = G.in_edges(node, data=True)
		ts = [edge[2]["timestamp"] for edge in edges]
		ts = numpy.array(sorted(ts))
	
	diff_ts = ts[1:] - ts[:-1]
	
	y, x =  numpy.histogram(diff_ts, bins=numpy.logspace(0, 8, num_iat_bins+1))
	y[0] = sum(numpy.array(diff_ts) == 0)
	if sum(y)!=0.0 and normalize:
		y = y*1.0/sum(y)
	#for i in range(1, len(y)):# In[1]:

	#y[i] += y[i-1]
	return x,y


def get_rating_dist(node, normalize):
	if "u" in node[0]:
		edges = G.out_edges(node, data=True)
		ts = [int(round(10*edge[2]["weight"])) for edge in edges]
	else:
		edges = G.in_edges(node, data=True)
		ts = [int(round(10*edge[2]["weight"])) for edge in edges]
   
	if NETWORKNAME in ["otc", "alpha"]: 
	   y, x =  numpy.histogram(ts, bins=range(-10, 12))
	else:
		y, x =  numpy.histogram(ts, bins=range(0, 6))

	if sum(y)!=0.0 and normalize:
		y = y*1.0/sum(y)
	return x,y


nodes = G.nodes()
# nodes = list(G.nodes)  # To solve AttributeError
edges = G.edges(data=True)
print ("%s network has %d nodes and %d edges" % (NETWORKNAME, len(nodes), len(edges)))

user_names = [node for node in nodes if "u" in node]
product_names = [node for node in nodes if "p" in node]
num_users = len(user_names)
num_products = len(product_names)
user_map = dict(zip(user_names, range(len(user_names))))
product_map = dict(zip(product_names, range(len(product_names))))

full_birdnest_user = pickle.load(open("../rev2data/%s/%s_birdnest_user.pkl" % (NETWORKNAME, NETWORKNAME), "rb"), encoding = 'latin1')
full_birdnest_product = pickle.load(open("../rev2data/%s/%s_birdnest_product.pkl" %(NETWORKNAME, NETWORKNAME), "rb"), encoding = 'latin1')

full_birdnest_edge = []
try:
	print ('loading birdnest pickle')
	full_birdnest_edge = pickle.load(open("../rev2data/%s/%s_edge_birdnest.pkl" % (NETWORKNAME, NETWORKNAME),"rb"), encoding = 'latin1')
	edge_map = pickle.load(open("../rev2data/%s/%s_edge_map.pkl" % (NETWORKNAME, NETWORKNAME),"rb"), encoding = 'latin1')
	full_birdnest_edge = numpy.array(full_birdnest_edge)
	mn = min(full_birdnest_edge)
	mx = max(full_birdnest_edge)
	full_birdnest_edge = (full_birdnest_edge - mn)*1.0/(mx-mn+0.001)
except:
	print ("Didnt find edge birdnest scores for %s network" % NETWORKNAME)
	full_birdnest_edge = [0.0]*len(edges)
	ae = zip(numpy.array(edges)[:,0], numpy.array(edges)[:, 1])
	edge_map = dict(zip(ae, range(len(edges))))

for node in nodes:
	if "u" in node[0]:
		G.node[node]["fairness"] = 1 - full_birdnest_user[user_map[node]]
	else:
		G.node[node]["goodness"] = (1 - full_birdnest_product[product_map[node]] - 0.5)*2

for edge in edges:
	G[edge[0]][edge[1]]["fairness"] = 1 - full_birdnest_edge[edge_map[(edge[0], edge[1])]]
iter = 0

yfg = []
ygood = []
xfg = []
du = 0
dp = 0
dr = 0

##### REV2 ITERATIONS START ######

# while iter < 500:
while iter < 1:
	print ('-----------------')
	print ("Epoch number %d with du = %f, dp = %f, dr = %f, for (%d,%d,%d,%d,%d,%d,%d)" % (iter, du, dp, dr, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3))
	if numpy.isnan(du) or numpy.isnan(dp) or numpy.isnan(dr):
			break
	
	du = 0
	dp = 0
	dr = 0
	
	############################################################

	print ('Updating goodness of product')

	currentgvals = []
	for node in nodes:
		if "p" not in node[0]:
			continue
		currentgvals.append(G.node[node]["goodness"])
	
	median_gvals = numpy.median(currentgvals) # Alternatively, we can use mean here, intead of median

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
	
	print ("Updating fairness of ratings")
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
		G.edge[edge[0]][edge[1]]["fairness"] = x
	
	############################################################
	
	currentfvals = []
	for node in nodes:
		if "u" not in node[0]:
			continue
		currentfvals.append(G.node[node]["fairness"])
		median_fvals = numpy.median(currentfvals) # Alternatively, we can use mean here, intead of median

	print ('updating fairness of users')
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

		mean_rating_fairness = numpy.mean(rating_fairness)

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
median_fvals = numpy.median(currentfvals)
print (len(currentfvals), median_fvals)

all_node_vals = []
for node in nodes:
	if "u" not in node[0]:
		continue
	f = G.node[node]["fairness"]
	all_node_vals.append([node, (f - median_fvals)*numpy.log(G.out_degree(node)+1), f, G.out_degree(node)])
all_node_vals = numpy.array(all_node_vals)
# all_node_vals = numpy.array(goodness_vals)

import operator
# sort users based on their scores
all_node_vals_sorted = sorted(all_node_vals, key= lambda x: (float(x[1]), float(x[2]), -1*float(x[3])))[::-1] 

fw = open("./results/%s-fng-sorted-users-%d-%d-%d-%d-%d-%d-%d.csv" % (NETWORKNAME, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3),"w")


badusers = []
goodusers = []	
for i, sl in enumerate(all_node_vals_sorted):
	# if sl[3] in badusers or sl[3] in goodusers: # dont store users for which we dont have ground truth
	fw.write("%s,%s,%s,%s\n" %(str(sl[0]), str(sl[1]), str(sl[2]), str(sl[3])))
fw.close()

