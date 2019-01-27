
import os
import sys
from sklearn.linear_model import LogisticRegression
from dumb_containers import evaluate_performance, evaluate_performance_sim
import numpy as np
import pdb

np.random.seed(666)

dataset = sys.argv[1]
emb_files = os.listdir('output/' + dataset)
embeddings = []

embed = []
nodes = []
for emb_file in emb_files:
    path = 'output/' + dataset + '/' + emb_file
    with open(path, 'r+') as f:
        f.readline() # skip the heading
        embed.append([])
        nodes.append([])
        for l in f:
            embed[-1].append([float(x) for x in l.split()[1:]])
            nodes[-1].append(l.split()[0])
                            
embed = np.array(embed)

'''
with open('data/' + dataset + '/' + 'label_os.csv', 'r') as f:
    label_eval = {x[0]: int(x[1]) 
        for x in map(lambda x: x.strip().split(), f)}
'''
with open('data/' + dataset + '/label_reg.csv', 'r') as f:
    label_reg = {x[0]: int(x[1]) 
        for x in map(lambda x: x.strip().split(), f)}

with open('data/' + dataset + '/label_os.csv', 'r') as f:
    label_os = {x[0]: int(x[1]) 
        for x in map(lambda x: x.strip().split(), f)}

import copy
label_eval = copy.deepcopy(label_reg)
for i, t in enumerate(label_os):
    label_eval[i] = t

train_set_ratio = 1.0 * len(label_reg) / len(label_eval)
print('num data: ' + str(len(label_eval)))
print('train set ratio: ' + str(train_set_ratio))

def generate_train_test():

    nodes_list = list(label_eval.keys())
    # pdb.set_trace()
    np.random.shuffle(nodes_list)
    train_set = nodes_list[:int(train_set_ratio * len(label_eval))]

    label_train = {x:label_eval[x] for x in train_set}
    label_test = {x:label_eval[x] for x in label_eval if x not in train_set}

    '''
    label_train = label_reg
    label_test = label_os
    '''
    # pdb.set_trace()
    train_sets = []
    for node, emb in zip(nodes, embed):
        X = []
        y = []
        for n, l in zip(node, emb):
            if n in label_train:
                y.append(label_train[n])
                X.append(l)
        train_sets.append((np.array(X),np.array(y)))

    test_sets = []
    for node, emb in zip(nodes, embed):
        X = []
        y = []
        for n, l in zip(node, emb):
            if n in label_test:
                y.append(label_test[n])
                X.append(l)
        test_sets.append((np.array(X),np.array(y)))
    print 1
    # pdb.set_trace()
    return train_sets, test_sets

def normalize(X):
    for i, t in enumerate(X):
        X[i] /= np.linalg.norm(t)

def sigmoid(X):
    return 1 / (1+np.exp(-X))

scores = []
repeat = 50

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,recall_score,precision_score, accuracy_score


for _ in range(repeat):
    train_sets, test_sets = generate_train_test()
    # pdb.set_trace()
    for (X_train, y_train), (X_test, y_test) in zip(train_sets, test_sets):

        # print X_train[0,:]; pdb.set_trace()

        fit = LogisticRegression()

        fit.fit(X_train, y_train)
        # y_pred = fit.predict(X_test)

        y_pred = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)

        accuracy_ = round(accuracy_score(np.array(y_test), y_pred), 2)
        recall_ = round(recall_score(np.array(y_test), y_pred, average='macro'), 2)
        precision_ = round(precision_score(np.array(y_test), y_pred, average='macro'), 2)
        f1_ = round(f1_score(np.array(y_test), y_pred, average='macro'), 2)

        scores.append(
            [accuracy_, recall_, precision_, f1_])

    if _%10==0:
        print str(_)+' repeats finish.'



scores = np.array(scores)
#scores_group = np.array([scores[i:i+repeat] for i in range(0,len(scores),repeat)])
scores_group = np.array([scores[list(range(i, len(scores), len(train_sets)))] for i in range(len(train_sets))])
scores_group = scores_group.mean(axis=1)

#print(emb_files)
#print(scores_group)
scores_group = [map(lambda x: np.round(x,2), t) for t in scores_group]
output = np.hstack(([[e] for e in emb_files], scores_group))

print('{0:40}: {1}'.format('method', ' '.join(['accuracy', 'recall', 'precision', 'f1'])))
for emb, scores in zip(emb_files, scores_group):
    print('{0:40}: {1}'.format(emb, ' '.join([str(x) for x in scores]).ljust(35)))

with open(dataset+'.csv', 'w+') as f:
    f.write('\n'.join([','.join(l) for l in output]))
    f.write('\n')
