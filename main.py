import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot azs plt
import time

from graph import Graph, load_edgelist, build_deepwalk_corpus, build_deepwalk_corpus_multi
import torch
import torch.nn as nn
import torch.nn.functional as F
from frame import Word2Vec

import sys

#import findspark
#findspark.init()
#from pyspark import SparkContext


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='corpus.txt', 
                output_file_name="output",
                min_count=1,
                iteration=15,
                margin=20,
                batch_size=64,
                emb_dimension=20)
    
    input_file_name="corpus.txt"
    min_count=1
    output_file_name="output"
    get_words(min_count)
    word_pair_catch = deque()
    init_sample_table()
    print('Word Count: %d' % len(self.word2id))
    print('Sentence Length: %d' % (self.sentence_length))
    
    
    
    
    
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
