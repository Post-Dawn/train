#!/usr/bin/env python
import numpy as np
import torch.optim as optim
import torch
import networkx as nx
import time
import argparse

from struc import construct_similarity_graph, get_sorted_degrees
from helpers import update_progress, normalize_graph, record_history
from model import all2vec
from train import train

# parameters.


def prepare_data(edgelist_url, label_url, K_layers, undirected):
    graph = nx.Graph()
    node_to_idx = {}
    idx_to_node = {}
    idx = 0

    with open(edgelist_url, 'r') as f:
        for l in f:
            l = l.split()
            if l[0] not in node_to_idx:
                node_to_idx[l[0]] = idx
                idx_to_node[idx] = l[0]
                idx+=1
            if l[1] not in node_to_idx:
                node_to_idx[l[1]] = idx
                idx_to_node[idx] = l[1]
                idx+=1
            graph.add_edge(node_to_idx[l[0]], node_to_idx[l[1]], weight=float(l[2]))
            if undirected:
                graph.add_edge(node_to_idx[l[1]], node_to_idx[l[0]], weight=float(l[2]))

    print('loaded %d edges and %d vertices' % (len(graph.edges()), len(graph.nodes())))

    print('reading the labels from ' + label_url)
    count = 0
    label_dict = {}
    with open(label_url, 'r') as f:
        for l in f:
            l = l.split()
            if l[0] in node_to_idx:
                count+=1
                label_dict[node_to_idx[l[0]]] = int(l[1])

    print('loaded %d labels' % len(label_dict))

    print('constructing structural similarity graphs...')
    print('sorting degree list...')
    sorted_deg_list = get_sorted_degrees(graph)
    print('sorting is done.')

    str_graph_edges = construct_similarity_graph(graph, K_layers, sorted_deg_list, True)[K_layers-1]
    str_graph = nx.Graph()
    for edge in str_graph_edges:
        str_graph.add_edge(edge[0], edge[1], weight=float(edge[2]))
        str_graph.add_edge(edge[1], edge[0], weight=float(edge[2]))

    print('normalizing graph edge weights...')
    graph = normalize_graph(graph)
    str_graph = normalize_graph(str_graph)

    edges = [[e[0], e[1], graph[e[0]][e[1]]['weight']] for e in graph.edges()]
    stredges = [[e[0], e[1], str_graph[e[0]][e[1]]['weight']] for e in str_graph.edges()]

    return list(graph.nodes()), edges, stredges, label_dict, node_to_idx, idx_to_node

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="An embedding algorithm. Input, output, label, and embedding dimension"
                                                 "must be specified.")
    parser.add_argument('--input', help='path to edge list file.')
    parser.add_argument('--output', help='path to embedding.')
    parser.add_argument('--label', help='path to labels.')
    parser.add_argument('--struct_order', '-k', help='the order of neighborhood to be used'
                                                     ' when computing structure similarity.'
                                                     'Default: 3.')
    parser.add_argument('--dimension', '-D', help='the embedding dimension.')
    parser.add_argument('--undirected', '-U', help='determines if the graph is constructed '
                                                   'in the undirected manner.')
    parser.add_argument('--margin', '-M', help='the margin used to separate discord pairs.'
                                               'Default: 10.')
    parser.add_argument('--epoch', help='number of training epochs. Default 12.')
    parser.add_argument('--batch_size', help='mini-batch size. Default 50.')
    parser.add_argument('--pre_epoch', help='number of pretraining epochs. Default 1/4 of '
                                            'the number of epochs.')

    parser.add_argument('--lr', '-r', help='learning rate. Default 0.025.')
    parser.add_argument('--cuda', '-c', help='determines if cuda is used.')
    parser.add_argument('--save_temps', help='if specified, save temp files '
                                             'in the directory. Default false.')

    parser.add_argument('--prob_r', help='probability to draw random pairs. Default: 0.3.')
    parser.add_argument('--prob_nbr', help='probability to draw a neighbor. Default: 0.5.')
    parser.add_argument('--neg_k', '-K', help='number of negative sampling. Default: 5.')

    args = parser.parse_args()
    if args.input is None:
        print('You must specify the input file.')
        exit()
    if args.output is None:
        print('You must specify the output ile.')
        exit()
    if args.label is None:
        print('You must specify the label file.')
        exit()
    if args.dimension is None:
        print('You must specify the embedding dimension.')
        exit()

    edgelist_url    = args.input
    label_url       = args.label
    output_url      = args.output
    n_dimension     = int(args.dimension)

    print('Input : ' + edgelist_url)
    print('Output: ' + output_url)
    print('Label : ' + label_url)

    params = {}
    params['prob_r']      = float(args.prob_r)   if args.prob_r     is not None else 0.3
    params['prob_nbr']    = float(args.prob_nbr) if args.prob_nbr   is not None else 0.5
    params['neg_k']       = int(args.neg_k)      if args.neg_k      is not None else 5
    params['lr']          = float(args.lr)       if args.lr         is not None else 0.025
    params['margin']      = int(args.margin)     if args.margin     is not None else 10
    params['batch_size']  = int(args.batch_size) if args.batch_size is not None else 50

    K_layers        = int(args.struct_order) if args.struct_order   is not None else 3
    undirected      = int(args.undirected)   if args.undirected     is not None else True
    num_epochs      = int(args.epoch)        if args.epoch          is not None else 12
    num_pre_epochs  = int(args.pre_epoch)    if args.pre_epoch      is not None else int(0.25 * num_epochs)
    use_cuda        = int(args.cuda)         if args.cuda           is not None else False
    save_temps      = str(args.save_temps)   if args.save_temps     is not None else None

    nodes_all, edges, stredges, label_dict, node_to_idx, idx_to_node  \
            = prepare_data(edgelist_url, label_url, K_layers, undirected)
    num_nodes = len(nodes_all)
    model = all2vec(num_nodes, n_dimension)

    # split train and test.
    train_set_ratio = 0.7
    len_train_set = int(train_set_ratio * len(label_dict))
    nodes_train = list(label_dict.keys())[:len_train_set]
    #nodes_train = np.random.choice(list(label_dict.keys()), len_train_set)

    print('making train-test split..')
    label_dict_train = {x: label_dict[x] for x in label_dict if x in nodes_train}
    print('number of training labels: %d' % len_train_set)
    print('train set positive ratio: %.2f' % (1.0 * sum(label_dict_train.values())/len(label_dict_train)))
    with open('train_set_emb.txt','w') as f:
        f.write('\n'.join([str(idx_to_node[x]) for x in label_dict_train.keys()]))

    '''
    edges_train = []
    stredges_train = []
    for e in edges:
        if e[0] in nodes_train or e[1] in nodes_train:
            edges_train.append(e)
    for e in stredges:
        if e[0] in nodes_train or e[1] in nodes_train:
            stredges_train.append(e)
    '''

    model = train(model, edges, stredges, label_dict_train,
                  params, use_cuda, num_epochs, num_pre_epochs, save_temps=save_temps)

    model.cpu()

    with open(output_url, 'w+') as f:
        for n in nodes_all:
            f.write(idx_to_node[n] + ' ' + ' '.join([str(x) for x in model.embedding(torch.tensor(n).long()).data.numpy()]))
            f.write('\n')
