
import time
import sys
import numpy as np
import torch

def prettytime(seconds):
    return seconds/3600, seconds/60%60, seconds%60

def update_progress(i, total, start_time, text=''):
    now = time.time()
    used = prettytime(now-start_time)
    eta = prettytime((now-start_time) / (i+1) * (total-i-1))
    output = ("\r%.2f%%, " % (100.0 * (i+1)/total) + 
                 "%d/%d, " % (i+1, total) + text + ': ' 
                 "used: %02d:%02d:%02d, eta: %02d:%02d:%02d" %
                (used[0], used[1], used[2],
                eta[0], eta[1], eta[2]))
    sys.stdout.write(output)
    sys.stdout.flush()
    if i == total-1:
        print('')

from copy import deepcopy
import math

def normalize_graph(g):
    new_g = deepcopy(g)
    degree_dict = {x: 0.0 for x in g.nodes()}
    for e in g.edges():
        degree_dict[e[0]] += g[e[0]][e[1]]['weight']
        degree_dict[e[1]] += g[e[0]][e[1]]['weight']

    for e in new_g.edges():
        new_g[e[0]][e[1]]['weight'] /= np.sqrt(degree_dict[e[0]] * degree_dict[e[1]] + 0.001)
    return new_g

def record_history(model, graph, str_graph, label, cuda=False, metric=np.linalg.norm):

    if cuda:
        embed = np.array([model.embedding(torch.tensor(n).cuda().long()).cpu().data.numpy() for n in range(len(graph))])
    else:
        embed = np.array([model.embedding(torch.tensor(n).long()).data.numpy() for n in range(len(graph))])
                            
    graph_connected_mean = np.mean([metric(embed[e[0]] - embed[e[1]]) for e in graph.edges()])
    str_connected_mean = np.mean([metric(embed[e[0]] - embed[e[1]]) for e in str_graph.edges()])
    graph_concord_mean = np.mean([metric(embed[e[0]] - embed[e[1]]) for e in graph.edges()
                                if e[0] in label and e[1] in label and label[e[0]] == label[e[1]]])
    str_concord_mean = np.mean([metric(embed[e[0]] - embed[e[1]]) for e in str_graph.edges()
                                if e[0] in label and e[1] in label and label[e[0]] == label[e[1]]])
    concord_mean =  np.mean([metric(embed[v]-embed[u]) for u in graph.nodes() for v in graph.nodes() 
                                 if u in label and v in label and label[u]==label[v]])
    disconcord_mean =  np.mean([metric(embed[v]-embed[u]) for u in graph.nodes() for v in graph.nodes() 
                                  if u in label and v in label and label[u]!=label[v]])
                                                  
    return graph_connected_mean, str_connected_mean, graph_concord_mean, str_concord_mean, concord_mean, disconcord_mean
