import numpy
from collections import deque

numpy.random.seed(12345)

def __init__(self, file_name, min_count):
        self.input_file_name = file_name
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))
        
def get_labels(self, label_file_name):
        with open(label_file_name, 'r') as f:
            self.label_dict = {self.word2id[x]: int(t) 
                               for x, t in map(lambda x: x.split(), f.readlines())
                               if x in self.word2id}
            
def get_random_pairs(self, n):
        pairs = numpy.random.choice(list(self.label_dict.keys()), size=(n,2))
        delta = list(map(lambda x: x[0]==x[1],
                [list(map(lambda x: self.label_dict[x], x)) for x in pairs]))
        return pairs, delta
    

def get_words(self, min_count):
        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0


        word_temp_1 = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_temp_1[w] += 1
                except:
                    word_temp_1[w] = 1
        word_temp_2=[]
        for key in word_temp_1:
            word_temp_2.append(tuple([int(key),word_temp_1[key]]))
        
        
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in word_temp_1.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)

def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        #print(self.word_frequency.values())
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        #print(pow_frequency)
        words_pow = sum(pow_frequency)
        #print(words_pow)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    # @profile
def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name)
                sentence = self.input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - window_size, 0):i + window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    if hasattr(self, 'label_dict') and u in self.label_dict and v in self.label_dict and self.label_dict[u] != self.label_dict[v]:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    # @profile
def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size
