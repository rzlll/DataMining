from collections import defaultdict, Counter
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import *
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from scipy.stats import entropy

def get_features(node):
    edges = G.out_edges(node, data=True)
    if len(edges) == 0:
        return -1

    if NETWORKNAME != "epinions":
        ts = [edge[2]["timestamp"] for edge in edges]
        ts = np.array(sorted(ts))
    else:
        ts = np.array([0,0,0,0,0])

    mnr = Counter(map(int, ts/(60*60*24.0))).most_common()[0][1]
    fnr = Counter(map(int, ts/(60*60*24.0))).most_common()[0][1]*1.0/len(edges)
    
    ratings = np.array([edge[2]["weight"] for edge in edges])
    pr = sum(ratings > 0)*1.0/len(ratings)
    nr = sum(ratings < 0)*1.0/len(ratings)

    avgRD = []
    for edge in edges:
        inrtgs = G.in_edges(edge[1], data=True)
        avgin = np.mean([e[2]["weight"] for e in inrtgs])
        avgRD.append(np.abs(edge[2]["weight"] - avgin))
    avgRD = np.mean(avgRD)

    BST = 0
    if abs(ts[0] - ts[-1]) < 60*60*24*28:
        BST =  1 - (abs(ts[0] - ts[-1]))*1.0/(60*60*24*28)

    elif ALGO == 15:
        return [avgRD] # SpamBehavior
    elif ALGO == 18:
        return [mnr, BST] # Spamicity
    elif ALGO == 20:
        return [mnr, pr, avgRD] # icwsm'13
    else:
        return []


NETWORKNAME = sys.argv[1]
ALGO = int(sys.argv[2])

print NETWORKNAME, ALGO

print "loading network"
G = cPickle.load(open("../network/%s_network.pkl" % (NETWORKNAME), "rb"))
print "done loading"

if NETWORKNAME == "otc" or NETWORKNAME == "alpha":
    allratings = np.arange(-10,11, 1)
else:
    allratings = np.arange(-10,11, 5)

print allratings

f = open("../network/%s_gt.csv" % NETWORKNAME,"r")
goodusers = set()
badusers = set()

for l in f:
        l = l.strip().split(",")
        if l[1] == "-1":
                badusers.add('u'+l[0])
        else:
                goodusers.add('u'+l[0])
f.close()
print "# badusers = ", len(badusers), "# goodusers = ", len(goodusers)

x = []
y = []
for good in goodusers:
    ft = get_features(good)
    if ft != -1:
        x.append(ft)
        y.append(1)

for bad in badusers:
    ft = get_features(bad)
    if ft != -1:
        x.append(ft)
        y.append(-1)

x, y = shuffle(x, y)
skf = StratifiedShuffleSplit(y, n_iter = 50, test_size = 1 - float(sys.argv[3]))
x = np.array(x)
y = np.array(y)

aucs = []
accs = []
for train, test in skf:

    train_x = x[train]
    test_x = x[test]
    train_y = y[train]
    test_y = y[test]
    print len(train_x), len(train_y), len(test_x), len(test_y)

    dt = RandomForestClassifier(n_estimators=100)
    dt.fit(np.array(train_x), train_y)

    #print np.argsort(dt.feature_importances_)
    
    pred_x = dt.predict_proba(test_x)[:, 1]
    
    #print zip(pred_x, test_y)
    
    print "AUC = ", roc_auc_score(test_y, pred_x),
    aucs.append(roc_auc_score(test_y, pred_x))
    
    pred_x = dt.predict(test_x)
    print "Accuracy = ", accuracy_score(test_y, pred_x)
    accs.append(accuracy_score(test_y, pred_x))
    
print "RESULT", sys.argv, np.mean(aucs), np.mean(accs)

