import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
#from frame import Word2Vec

import sys

#import findspark
#findspark.init()
#from pyspark import SparkContext

    
word2id = dict()
id2word = dict()
word_frequency = dict()
word_count=0
sentence_length = 0
sentence_count = 0
sample_table = []

def init_sample_table():
    global sample_table
    global word_frequency
    sample_table_size = 1e8
    pow_frequency = numpy.array(list(word_frequency.values()))**0.75
    words_pow = sum(pow_frequency)
    ratio = pow_frequency / words_pow
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


def evaluate_pair_count(window_size):
    return sentence_length * (2 * window_size - 1) - (
        sentence_count - 1) * (1 + window_size) * window_size

if __name__ == '__main__':
    input_file_name="corpus.txt"
    min_count=1
    output_file_name="output"
    get_words(input_file_name,min_count)
    word_pair_catch = deque()
    init_sample_table()
    
    print('Word Count: %d' % len(word2id))
    print('Sentence Length: %d' % (sentence_length))

    pos_pairs = get_batch_pairs(64, 5)
    print(pos_pairs)
    '''emb_size = len(word2id)
    emb_dimension = 20
    batch_size = 64
    window_size = 5
    iteration = 15
    initial_lr = initial_lr
    skip_gram_model = SkipGramModel(emb_size, emb_dimension)
    margin = 20
    optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)

    pair_count = evaluate_pair_count(window_size)
    batch_count = int(iteration * pair_count / batch_size)


    loss_avg = 0.0
    loss_r_pos = 0.0
    loss_r_neg = 0.0


    w2v.train()'''