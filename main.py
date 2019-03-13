import numpy
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import SkipGramModel
#from frame import Word2Vec

import sys

import findspark
findspark.init()
from pyspark import SparkContext

    
word2id = dict()
id2word = dict()
word_frequency = dict()
word_count=0
sentence_length = 0
sentence_count = 0
sample_table = []
batch_size=0
window_size=0
emb_dimension = 0
emb_size = 0

def init_sample_table():
    global sample_table
    global word_frequency
    sample_table_size = 1e8
    pow_frequency = np.array(list(word_frequency.values()))**0.75
    words_pow = sum(pow_frequency)
    ratio = pow_frequency / words_pow
    #print(ratio*sample_table_size)
    count = numpy.round(ratio * sample_table_size)
    for wid, c in enumerate(count):
        sample_table += [wid] * int(c)
    sample_table = numpy.array(sample_table)

def get_words(input_file_name,min_count):
    global sentence_count
    global sentence_length
    input_file = open(input_file_name)
    word_temp_1 = dict()
    for line in input_file:
        sentence_count += 1
        line = line.strip().split(' ')
        sentence_length += len(line)
        for w in line:
            try:
                word_temp_1[w] += 1
            except:
                word_temp_1[w] = 1
    word_temp_2=[]
    for key in word_temp_1:
        word_temp_2.append(tuple([int(key),word_temp_1[key]]))
        
    #print(sentence_length)
    wid = 0
    global word_frequency
    global word_count
    for w, c in word_temp_1.items():
        if c < min_count:
            sentence_length -= c
            continue
        word2id[w] = wid
        id2word[wid] = w
        word_frequency[wid] = c
        wid += 1
    word_count = len(word2id)

def get_batch_pairs(batch_size, window_size):
    global word_pair_catch
    input_file = open(input_file_name)
    while len(word_pair_catch) < batch_size:
        sentence = input_file.readline()
        if sentence is None or sentence == '':
                input_file = open(input_file_name)
                sentence = self.input_file.readline()
        word_ids = []
        for word in sentence.strip().split(' '):
            try:
                word_ids.append(word2id[word])
            except:
                continue
        for i, u in enumerate(word_ids):
            for j, v in enumerate(
                    word_ids[max(i - window_size, 0):i + window_size]):
                assert u < word_count
                assert v < word_count
                if i == j:
                    continue
                word_pair_catch.append((u, v))
    batch_pairs = []
    for _ in range(batch_size):
        batch_pairs.append(word_pair_catch.popleft())
    return batch_pairs

def get_batch(window_size):
    global word_pair_catch
    input_file = open(input_file_name)
    while len(word_pair_catch)==0:
        sentence = input_file.readline()
        if sentence is None or sentence == '':
                #input_file = open(input_file_name)
                #sentence = self.input_file.readline()
                break
        word_ids = []
        for word in sentence.strip().split(' '):
            try:
                word_ids.append(word2id[word])
            except:
                continue
        for i, u in enumerate(word_ids):
            for j, v in enumerate(
                    word_ids[max(i - window_size, 0):i + window_size]):
                assert u < word_count
                assert v < word_count
                if i == j:
                    continue
                word_pair_catch.append((u, v))
    now=word_pair_catch.popleft()
    #print(now)
    return now

def get_neg_v_neg_sampling(pos_word_pair, count):
    global sample_table
    neg_v = numpy.random.choice(
        sample_table, size=(len(pos_word_pair), count)).tolist()
    return neg_v

def evaluate_pair_count(window_size):
    return sentence_length * (2 * window_size - 1) - (
        sentence_count - 1) * (1 + window_size) * window_size

def plus(x):
    '''now1=[]
    for i in x:
        now2=[]
        for j in i:
            now2.append(j+1)
        now1.append(tuple(now2))
    return tuple(now1)'''
    return 1

def matrix2rdd(matrix1,matrix2,num_batch):
    global batch_size,window_sizesentence_length
    origin = []    
    for i in matrix1:
        origin.append(i)
    for i in matrix2:
        origin.append(i)

    origin_set=[]
    for i in range(num_batch):
        origin_now = list(origin)
        pos_pairs = get_batch(window_size)
        pos_u = pos_pairs[0]
        pos_v = pos_pairs[1]
        origin_now.append(pos_u)
        origin_now.append(pos_v)
        origin_now=tuple(origin_now)    
        origin_set.append(origin_now)
    return origin_set


def transform(x):
    import numpy as np
    now = list (x)
    u = now[0:emb_size]
    v = now[emb_size:2*emb_size]

    u=np.array(u)
    v=np.array(v).T

    pos_u = now[2*emb_size]
    pos_v = now[2*emb_size+1]

    hidden = np.array(u[pos_u])# 1*dimension
    y_pred = np.dot(hidden,v) # 1*dimension
    y_exp=np.exp(y_pred)
    y_exp_sum = y_exp.sum(axis=0)
    y_softmax = y_exp / y_exp_sum
    y_softmax[pos_v]=y_softmax[pos_v]-1

    d2 = np.outer( hidden, y_softmax)
    d1 = np.dot(d2,y_softmax)

    for i in range(emb_dimension):
        u[pos_u][i]=u[pos_u][i]+d1[i]
    
    for i in range(emb_dimension):
        for j in range(emb_size):
            v[i][j]=v[i][j]+d2[i][j]
    v=v.T
    result = []
    result.append(u.tolist())
    result.append(v.tolist())

    return tuple(result)

if __name__ == '__main__':
    sc=SparkContext()
    input_file_name="corpus.txt"
    min_count=1
    output_file_name="output"
    get_words(input_file_name,min_count)
    word_pair_catch = deque()
    init_sample_table()
    
    print('Word Count: %d' % len(word2id))
    print('Sentence Length: %d' % (sentence_length))

    '''pos_pairs = get_batch_pairs(64, 5)
    neg_v = get_neg_v_neg_sampling(pfrom torch.autograd import Variableos_pairs, 5)'''
    #print(pos_pairs)
    
    emb_size = len(word2id)
    #print(emb_size)
    emb_dimension = 20
    batch_size = 64
    window_size = 5
    iteration = 15
    '''initial_lr = initial_lr
    skip_gram_model = SkipGramModel(emb_size, emb_dimension)
    margin = 20
    optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)'''

    pair_count = evaluate_pair_count(window_size)
    batch_count = int(iteration * pair_count / batch_size)
    
    loss_avg = 0.0
    loss_r_pos = 0.0
    loss_r_neg = 0.0

    num_batch=10 
    '''if(batch_count/num_batch==0):
        total_cycle=batch_count/num_batch
    else:
        total_cycle=(int)(batch_count/num_batch)+1'''
    total_cycle=1

    initrange = 0.5 / emb_dimension

    origin_mid_u = np.random.uniform(-1,1,(emb_size,emb_dimension)).tolist()
    origin_mid_v = np.random.uniform(-1,1,(emb_size,emb_dimension)).tolist()

    origin_matrix = matrix2rdd(origin_mid_u,origin_mid_v, num_batch)
    
    #for i in range()
    origin_matrix_rdd = sc.parallelize(origin_matrix)
    for i in range(total_cycle):
        matrix=origin_matrix_rdd.map(transform).collect()
        mid_u = np.zeros((emb_size, emb_dimension))
        mid_v = np.zeros((emb_size, emb_dimension))
        for i in range(num_batch):
            mid_result_u = np.array(matrix[i][0])
            mid_result_v = np.array(matrix[i][1])
            mid_u = mid_u + mid_result_u
            mid_v = mid_v + mid_result_v
        mid_u = mid_u / num_batch
        mid_v = mid_v / num_batch
        origin_matrix = matrix2rdd(mid_u, mid_v, num_batch)
        origin_matrix_rdd = sc.parallelize(origin_matrix)
    print(mid_u)

    #origin_u=tuple(origin_u)
    #print(origin_u)

    #print(bb)

    '''for i in range(emb_size)

    for i in range(int(batch_count/num_batch)):'''