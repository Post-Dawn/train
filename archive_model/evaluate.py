
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from dumb_containers import evaluate_performance

input_emb = sys.argv[1]
input_label = sys.argv[2]
input_train_emb = sys.argv[3]

print('Loading embedding: %s\n Loading label: %s' % (input_emb, input_label))
print('Loading train emb: %s' % input_train_emb)

with open(input_emb,'r') as f:
    emb = np.array([[float(x) for x in y.split()] for y in f])

with open(input_label,'r') as f:
    label_dict = {float(x[0]): int(x[1]) for x in map(lambda x: x.split(), f.readlines())}

with open(input_train_emb, 'r') as f:
    train_set_emb = [float(x) for x in f.readlines()]

nodes = list(set(emb[:, 0]))
node_to_idx = {t:x for x, t in enumerate(nodes)}
emb = emb[:, 1:]
for i, t in enumerate(emb):
    emb[i] /= np.linalg.norm(t)
nodes_labeled = [x for x in nodes if x in label_dict]
nodes_used = [x for x in nodes_labeled if x in train_set_emb]
nodes_unused = [x for x in nodes_labeled if x not in train_set_emb]

print('len of labeled nodes: %d' % len(nodes_labeled))
print('len of unused nodes: %d' % len(nodes_unused))

nodes_train = np.random.choice(nodes_unused, int(0.7 * len(nodes_unused)))
nodes_test = list(set(nodes_unused) - set(nodes_train))

print('num of train set: %d' % len(nodes_train))
print('num of test set: %d' % len(nodes_test))

X_train = np.array([emb[node_to_idx[x]] for x in nodes_used])
y_train = np.array([label_dict[x] for x in nodes_used])
X_test = np.array([emb[node_to_idx[x]] for x in nodes_test])
y_test = np.array([label_dict[x] for x in nodes_test])
#X_test = X_train
#y_test = y_train

print('train pos rate: %f ' % (1.0 * sum(y_train) / len(y_train)))
print('test pos rate: %f ' % (1.0 * sum(y_test) / len(y_test)))

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_test_score = X_test.dot(lr.coef_.T) + lr.intercept_
y_test_score = y_test_score.flatten()

print(evaluate_performance(y_test, y_test_score, verbose=True))
plt.show()


