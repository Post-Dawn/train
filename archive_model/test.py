
import numpy as np

def get_labeled_edges(edges, label_dict):
    new_edges = []
    for e in edges:
        if e[0] in label_dict and e[1] in label_dict:
            new_edges.append(e)

    return new_edges


def get_inter_edges(edges, label_dict, label):
    new_edges = []
    for e in edges:
        src = e[0]
        tar = e[1]
        if (src in label_dict and tar not in label_dict
            and label_dict[src] == label):
            new_edges.append(e)
        if (tar in label_dict and src not in label_dict
            and label_dict[tar] == label):
            new_edges.append(e)

    return new_edges


def get_unsup_edges(edges, label_dict):
    new_edges = []
    for e in edges:
        src = e[0]
        tar = e[1]
        if src not in label_dict and tar not in label_dict:
            new_edges.append(e)
    return new_edges


with open('graph/edgelist.csv', 'r') as f:
    edges = [[float(x) for x in y.split()] for y in f.readlines()]

with open('graph/label.csv', 'r') as f:
    label_dict = {float(x.split()[0]): int(x.split()[1]) for x in f.readlines()}


train_set_ratio = 0.4
len_train_set = int(train_set_ratio * len(label_dict))
nodes_train = list(label_dict.keys())[:len_train_set]

print('making train-test split..')
label_dict_train = {x: label_dict[x] for x in label_dict if x in nodes_train}

print(len(get_labeled_edges(edges, label_dict_train)),
            len(get_inter_edges(edges, label_dict_train, 1)), 
            len(get_inter_edges(edges, label_dict_train, 0)), 
            len(get_unsup_edges(edges, label_dict_train)))

print(sum([len(get_labeled_edges(edges, label_dict_train)),
            len(get_inter_edges(edges, label_dict_train, 1)), 
            len(get_inter_edges(edges, label_dict_train, 0)), 
            len(get_unsup_edges(edges, label_dict_train))]))

print(len(edges))
