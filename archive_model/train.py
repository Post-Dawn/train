import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from model import all2vec

from helpers import update_progress, normalize_graph
from sampling import *
import time


def n_mean(arr):
    if len(arr)==0:
        return float('nan')
    else:
        return np.mean(arr)


def get_concord_labeled(edges, label):
    labeled = []
    concord = []
    for e in edges:
        if e[0] in label and e[1] in label:
            labeled.append(1)
        else:
            labeled.append(0)
        if e[0] in label and e[1] in label and label[e[0]] != label[e[1]]:
            concord.append(0)
        else:
            concord.append(1)
    return np.array(concord), np.array(labeled)


def normalize_embedding(embedding, index):
    embed_weight = torch.tensor(embedding.weight)
    embed_weight[index]  = F.normalize(embed_weight[index])
    embedding.weight = nn.Parameter(embed_weight)


def train(model, edges, stredges, label, params, use_cuda, num_epoch, num_pre_epoch, save_temps=None):
    if 'prob_r' not in params:
        prob_r = 0.2
    else:
        prob_r = params['prob_r']

    if 'prob_nbr' not in params:
        prob_nbr = 0.5
    else:
        prob_nbr = params['prob_nbr']

    if 'neg_k' not in params:
        neg_k = 5
    else:
        neg_k = params['neg_k']

    if 'margin' in params:
        margin = params['margin']
    else:
        margin = 10

    if 'lr' in params:
        lr = params['lr']
    else:
        lr = 0.025

    if 'batch_size' in params:
        batch_size = params['batch_size']
    else:
        batch_size = 50

    train_per_epoch = len(edges) + len(stredges)

    nodes_all = list(set(np.array(edges)[:, :2].flatten()))
    nodes_labeled = [n for n in nodes_all if n in label]

    edges_np = edges
    stredges_np = stredges
    edges = torch.tensor(edges).float()
    stredges = torch.tensor(stredges).float()


    #concord, labeled = get_concord_label(edges, label)
    #strconcord, strlabeled = get_concord_label(stredges, label)

    print('sampling negative edges...')
    negs_all = np.array(np.random.choice(nodes_all, num_epoch * neg_k * train_per_epoch))
    negs_all.resize(train_per_epoch * num_epoch, neg_k)
    negs_all = torch.tensor(negs_all).long()

    if torch.cuda.is_available() and use_cuda:
        model.cuda()
        negs_all = negs_all.cuda()
        edges = edges.cuda()
        stredges = stredges.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr)

    # the following variables are for recording training process.
    it_nbr = 0
    it_str = 0
    it_lbl = 0
    counter = 0
    embedding_history = {}
    mean_history = []
    score_history = []

    emb_save_epoch = 10

    tmp_loss = []
    '''
    tmp_loss_1_pos = []
    tmp_loss_2_pos = []
    tmp_loss_1_neg = []
    tmp_loss_2_neg = []
    '''
    tmp_loss_1_pos = 0.0
    tmp_loss_2_pos = 0.0
    tmp_loss_1_neg = 0.0
    tmp_loss_2_neg = 0.0
    avg_interval = 50

    penalty=1

    print('--------------------------------')
    print('# Nodes : %d' % len(nodes_all))
    print('# Labels: %d' % len(label))
    print('# Edges : %d' % len(edges))
    print('# Str Pairs: %d' % len(stredges))

    print('# Training epochs: %d\n# Pre-training   : %d' % (num_epoch, num_pre_epoch))
    print('Mini-batch size  : %d' % batch_size)
    if use_cuda and torch.cuda.is_available():
        print('Using cuda.')

    print('--------------------------------')
    print('Parameters:')
    print('Learning rate        : %f' % lr)
    print('Embedding dimension  : %d'% model.n_dimension)
    print('Negative sampling    : %d' % params['neg_k'])
    print('Hard margin          : %f' % params['margin'])
    print('Probability random   : %f' % prob_r)
    print('Probability neighbor : %f'% prob_nbr)
    print('--------------------------------')
    print('Begin Training.')

    nodes_pos = list([x for x in label if label[x] == 1])
    nodes_neg = list([x for x in label if label[x] == 0])
    new_embed = torch.tensor(model.embedding.weight)
    new_embed[nodes_pos] = torch.abs(new_embed[nodes_pos])
    new_embed[nodes_neg] = -torch.abs(new_embed[nodes_neg])
    model.embedding.weight = nn.Parameter(new_embed)

    print(len(nodes_pos), len(nodes_neg))

    start = time.time()
    labeled_edges = np.array(get_labeled_edges(edges_np, label))
    inter_edges_neg = np.array(get_inter_edges(edges_np, label, 0))
    inter_edges_pos = np.array(get_inter_edges(edges_np, label, 1))
    unsup_edges = np.array(get_unsup_edges(edges_np, label))
    
    print(len(labeled_edges), len(inter_edges_neg), len(inter_edges_pos), len(unsup_edges))

    num_random = int(prob_r * train_per_epoch)
    num_det = train_per_epoch - num_random
    num_labeled = int(0.2 * num_det)
    num_inter = int(0.6 * num_det)
    num_unsup = int(0.2 * num_det)

    num_res = train_per_epoch - (num_labeled + num_inter + num_unsup + num_random)
    num_random += num_res

    for i in range(num_epoch):
        #s_time = time.time()
        #if i >= num_pre_epoch:
        #    normalize_embedding(model.embedding, list(label.keys()))
        #    normalize_embedding(model.embedding_context, list(label.keys()))
        # generate a batch of random edges for this epoch.
        random_edges = np.array(np.random.choice(nodes_labeled, num_random * 2))
        random_edges.resize(int(len(random_edges)/2), 2)
        random_edges_np = np.hstack((random_edges, penalty * np.ones((len(random_edges), 1))))
        labeled_edges_epoch = labeled_edges[np.random.choice(len(labeled_edges), num_labeled)]
        inter_edges_pos_epoch = inter_edges_pos[np.random.choice(len(inter_edges_pos), int(0.3 * num_inter))]
        inter_edges_neg_epoch = inter_edges_neg[np.random.choice(len(inter_edges_neg), int(0.7 * num_inter))]
        unsup_edges_epoch = unsup_edges[np.random.choice(len(unsup_edges), num_unsup)]

        g_edges = np.vstack( (random_edges_np, labeled_edges_epoch,
                            inter_edges_pos_epoch, inter_edges_neg_epoch,
                            unsup_edges_epoch) )

        np.random.shuffle(g_edges) 
        concord, is_labeled = get_concord_labeled(g_edges, label)

        g_edges = torch.tensor(g_edges).float()
        concord = torch.tensor(concord).float()
        is_labeled = torch.tensor(is_labeled).float()
        if use_cuda:
            g_edges = g_edges.cuda()
            concord = concord.cuda()
            is_labeled = is_labeled.cuda()
        #print('sampling used %.2f' % (time.time() - s_time))

        # increasing penalty seem to make the scores -> infinity after several epochs
        # penalty = float(np.power(10, i/10))
        if i == num_pre_epoch - 1:
            print('')
            print('pre-training ended after %d epochs' % (i+1))

        negs = negs_all[i*(train_per_epoch): (i+1) * train_per_epoch]
        for it in range(int(train_per_epoch/batch_size)):
            optimizer.zero_grad()

            edges_batch = g_edges[it * batch_size: min((it+1) * batch_size, train_per_epoch)]
            delta_batch = concord[it * batch_size: min((it+1) * batch_size, train_per_epoch)]
            neg_batch = negs[it * batch_size: min((it+1) * batch_size, train_per_epoch)]
            is_label_batch = is_labeled[it * batch_size : min((it+1) * batch_size, train_per_epoch)]
            if len(edges_batch) != len(neg_batch):
                print('\nerror! inconsistent batch size')
                print('expected start: %d' % (it*batch_size))
                print('expected end: %d' % (min((it+1) * batch_size, train_per_epoch)))
                print(g_edges.size())
                print(edges_batch.size())
                print(neg_batch.size())
                print(negs[it * batch_size: min((it+1) * batch_size, train_per_epoch)].size())
            score1_all, score2_all = model.forward(edges_batch, neg_batch)

            if i < num_pre_epoch:
                score1_all = torch.mul(score1_all, is_label_batch)
                score2_all = torch.mul(score2_all, is_label_batch)

            score1_pos = torch.sum(torch.mul(delta_batch, score1_all))
            score1_neg = torch.sum(torch.clamp(margin - torch.mul(1-delta_batch, score1_all), min=0))
            score2_pos = torch.sum(torch.mul(delta_batch, score2_all))
            score2_neg = torch.sum(torch.clamp(margin - torch.mul(1-delta_batch, score2_all), min=0))
            loss = score1_pos + score1_neg + score2_pos + score2_neg
            loss.backward()
            optimizer.step()
            counter += 1

            nodes_pos_updated = edges_batch.cpu().data.numpy().flatten()
            nodes_neg_updated = neg_batch.cpu().data.numpy().flatten()
            nodes_updated = list(set(nodes_pos_updated).union(nodes_neg_updated))
            normalize_embedding(model.embedding, nodes_updated)

            avg_exp = 0.95
            num_pos = torch.sum(delta_batch).item()
            num_neg = torch.sum(1-delta_batch).item()
            if num_pos > 0:
                score1_loss_pos = score1_pos.item() / num_pos
                score2_loss_pos = score2_pos.item() / num_pos
                tmp_loss_1_pos = avg_exp * tmp_loss_1_pos + (1-avg_exp) * score1_loss_pos
                tmp_loss_2_pos = avg_exp * tmp_loss_2_pos + (1-avg_exp) * score2_loss_pos

            if num_neg > 0:
                score1_loss_neg = torch.sum(torch.mul(
                    1-delta_batch, score1_all)).item() / torch.sum(1-delta_batch).item()
                score2_loss_neg = torch.sum(torch.mul(
                    1-delta_batch, score2_all)).item() / torch.sum(1-delta_batch).item()
                tmp_loss_1_neg = avg_exp * tmp_loss_1_neg + (1-avg_exp) * score1_loss_neg
                tmp_loss_2_neg = avg_exp * tmp_loss_2_neg + (1-avg_exp) * score2_loss_neg

            update_progress(it * batch_size + i * train_per_epoch,
                            num_epoch * train_per_epoch,
                            start, 'pos 1: %2.2f, pos 2: %2.2f, neg 1: %2.2f, neg 2: %2.2f ' %
                            (tmp_loss_1_pos, tmp_loss_2_pos,
                            tmp_loss_1_neg, tmp_loss_2_neg))

        if save_temps is not None:
            if i % emb_save_epoch == emb_save_epoch-1:
                if use_cuda:
                    embedding_history[i] = np.array([model.embedding(torch.tensor(n).long().cuda()).cpu().data.numpy()
                                                     for n in range(len(nodes_all))])
                else:
                    embedding_history[i] = np.array([model.embedding(torch.tensor(n).long()).cpu().data.numpy()
                                                     for n in range(len(nodes_all))])
                if save_temps[-1] == '/':
                    save_temps = save_temps[:-1]
                save_tmp_path = save_temps + '/embedding_' + str(i)
                with open(save_tmp_path, 'w+') as f:
                    f.write('\n'.join([' '.join([str(x) for x in l]) for l in embedding_history[i]]))

    return model

