import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from six import iterkeys

import torch
import torch.nn as nn
import torch.nn.functional as F
from frame import Word2Vec
import random
from random import shuffle

import sys

import findspark
findspark.init()
from pyspark import SparkContext

def remove_self_loops(G):
    removed = 0
    for x in G:
        if x in G[x]: 
            G[x].remove(x)
            removed += 1

#    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
#    return self

def make_consistent(G):
    for k in iterkeys(G):
        G[k] = list(sorted(set(G[k])))
    remove_self_loops(G)

def load_edgelist(G,file_, undirected=True, base=10):
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x, base)
            y = int(y, base)
            if(not(x in G)):
                G[x]=[]
            G[x].append(y)
            if undirected:
                if(not(y in G)):
                    G[y]=[]
                G[y].append(x)
    make_consistent(G)

def build_deepwalk_corpus(G, path_G, num_paths, path_length, alpha=0,rand=random.Random(0)):
    nodes=[]
    for i in G:
        nodes.append(i)
    #print(nodes)
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        #for i in nodes:
        #    path_G.append(i)
        path_G.extend(nodes)
    #print(path_G)

def build_deepwalk_corpus_multi(G, G_str, path_G, path_G_str, prob_str, num_paths, path_length, alpha=0,rand=random.Random(0)):
    nodes=[]
    for i in G:
        nodes.append(i)
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            dice = rand.random()
            if dice <= prob_str and node in G_str and len(G_str[node]) > 0:
                path_G_str.append(node)
            else:
                path_G.append(node)
                
def G_random_walk(x):
    import random
    rand=random.Random()
    #if x:
    path = [x]
    #else:
    #    path = [rand.choice(list(G.keys()))]
    while len(path) < path_length:
        cur = path[-1]
        if rand.random() >= 0:
            if len(G[cur]) > 0:
                path.append(rand.choice(G[cur]))
            else:
                path.append(path[0])
        else:
            break
    return path

def G_str_random_walk(x):
    #import random
    rand=random.Random()
    #if x:
    path = [x]
    #else:
    #    path = [rand.choice(list(G.keys()))]
    while len(path) < path_length:
        cur = path[-1]
        if rand.random() >= 0:
            if len(G_str[cur]) > 0:
                path.append(rand.choice(G[cur]))
            else:
                path.append(path[0])
        else:
            break
    return path


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        label_file = sys.argv[3]
    except:
        label_file = None
    G={}
    G_str={}
    path_G=[]
    path_G_str=[]
    if '--margin' in sys.argv:
        margin = int(sys.argv[sys.argv.index('--margin')+1])
    else:
        margin=20
    sc= SparkContext()
    load_edgelist(G,input_file, base = 10)
    num_paths=10
    path_length=10
    if '--str' in sys.argv:
        path_to_str = sys.argv[sys.argv.index('--str')+1]
        load_edgelist(G_str,path_to_str, base=10)
        print(len(G_str), len(G))
        if '--prob_str' in sys.argv:
            prob_str = float(sys.argv[sys.argv.index('--prob_str')+1])
        else:
            prob_str = 0.3
        print('building corpus with structural graph in ' + path_to_str)
        print('with probability ' + str(prob_str))
        path_length=5
        build_deepwalk_corpus_multi(G, G_str, path_G, path_G_str, prob_str, 10, 5)
    else:
        build_deepwalk_corpus(G, path_G, num_paths=10, path_length=10)
    #print(path_G)
    G_rdd=sc.parallelize(path_G)
    G_str_rdd=sc.parallelize(path_G_str)
    #random_rdd=sc.parallelize([1,2])
    #random_rdd=random_rdd.map(lambda x : random_walk(G,x))
    G_rdd=G_rdd.map(G_random_walk)
    G_str_rdd=G_str_rdd.map(G_str_random_walk)
    deep_G=G_rdd.collect()
    deep_G_str=G_str_rdd.collect()
    #with open('corpus.txt', 'w+') as f:
    f = open("corpus.txt", "w+")
    for now in deep_G:
        count=0
        for x in now:
            count=count+1
            f.write("%d"%x)
            if(count!=10):
                f.write(" ")
        f.write("\n")
    f.close()
