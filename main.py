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
    for w, c in word_temp_1.items():
        if c < min_count:
            sentence_length -= c
            continue
        word2id[w] = wid
        id2word[wid] = w
        word_frequency[wid] = c
        wid += 1
    word_count = len(word2id)


if __name__ == '__main__':
    '''w2v = Word2Vec(input_file_name='corpus.txt', 
                output_file_name="output",
                min_count=1,
                iteration=15,
                margin=20,
                batch_size=64,
                emb_dimension=20)'''


    input_file_name="corpus.txt"
    min_count=1
    output_file_name="output"
    get_words(input_file_name,min_count)
    word_pair_catch = deque()
    init_sample_table()
    
    print('Word Count: %d' % len(word2id))
    print('Sentence Length: %d' % (sentence_length))
    
    
    
    
    
    '''output_file_name = output_file_name
    emb_size = len(self.data.word2id)
    emb_dimension = emb_dimension
    batch_size = batch_size
    window_size = window_size
    iteration = iteration
    initial_lr = initial_lr
    skip_gram_model = SkipGramModel(emb_size, emb_dimension)
    margin = margin
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        skip_gram_model.cuda()
    optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)

    label_file=None
    if label_file is not None:
        data.get_labels(label_file)
    
    w2v.train()'''
