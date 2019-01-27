import time

import findspark
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from helpers import update_progress
from input_tool import InputData
from model import SkipGramModel
from pyspark import SparkContext

findspark.init()
sc= SparkContext()

def train(data, min_count, iteration, margin, batch_size, emb_dimension):
        pair_count = data.evaluate_pair_count(window_size)
        batch_count = int(iteration * pair_count / batch_size)

        #process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)

        '''
        pos_pairs_all = self.data.get_batch_pairs(int(self.batch_size * batch_count), self.window_size)
        neg_v_all = self.data.get_neg_v_neg_sampling(pos_pairs_all, 5)
        pos_u_all = [pair[0] for pair in pos_pairs_all]
        pos_v_all = [pair[1] for pair in pos_pairs_all]

        pos_u_all = Variable(torch.LongTensor(pos_u_all))
        pos_v_all = Variable(torch.LongTensor(pos_v_all))
        neg_v_all = Variable(torch.LongTensor(neg_v_all))
        '''
        
        n_rand = 40
        
        if not hasattr(data, 'label_dict'):
            use_label = False
        else:
            use_label = True
            print('using labels')
            print('margin: ' + str(margin))

        
        if use_label:
            random_pairs_all, delta = data.get_random_pairs(int(n_rand * batch_count))
            random_pairs_all = Variable(torch.LongTensor(random_pairs_all))
            delta = Variable(torch.FloatTensor(delta))

        if self.use_cuda:
            '''
            pos_u_all = pos_u_all.cuda()
            pos_v_all = pos_v_all.cuda()
            neg_v_all = neg_v_all.cuda()
            '''
            if use_label:
                delta = delta.cuda()
                random_pairs_all = random_pairs_all.cuda()

        #process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        #for i in process_bar:

        loss_avg = 0.0
        loss_r_pos = 0.0
        loss_r_neg = 0.0
        start = time.time()
        for i in range(int(batch_count)):
            pos_pairs = data.get_batch_pairs(batch_size,window_size)
            #print(pos_pairs)
            #time.sleep(10)
            neg_v = data.get_neg_v_neg_sampling(pos_pairs, 5)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))
                                                      
            if self.use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()

            '''
            pos_from = i * self.batch_size
            pos_to = (i+1) * self.batch_size
            pos_u = pos_u_all[pos_from:pos_to]
            pos_v = pos_v_all[pos_from:pos_to]
            neg_v = neg_v_all[pos_from:pos_to]
            '''
            
            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()
            loss_avg = 0.95 * loss_avg + 0.05 * loss.item() / self.batch_size
            
            if use_label:
                r_pos_from = i * n_rand
                r_pos_to = (i+1) * n_rand
                rand_pairs = random_pairs_all[r_pos_from:r_pos_to]
                det = delta[r_pos_from:r_pos_to]
                r_neg_v = self.data.get_neg_v_neg_sampling(rand_pairs, 5)
                r_neg_v = Variable(torch.LongTensor(r_neg_v))
                if self.use_cuda:
                    r_neg_v = r_neg_v.cuda()


                self.optimizer.zero_grad()
                loss_rand = self.skip_gram_model.det_forward(rand_pairs[:,0],rand_pairs[:,1], r_neg_v)
                #print(loss_rand)
                #return
                #print(det.size(),loss_rand.size(), r_neg_v.size())
                #loss_rand_pos = torch.sum(torch.mul(det, loss_rand))
                loss_rand_pos = torch.sum(torch.clamp(det * loss_rand - self.margin, min=0))
                loss_rand_neg = torch.sum(torch.clamp(self.margin - torch.mul(1-det, loss_rand), min=0))

                loss_rand_sum = loss_rand_pos + loss_rand_neg
                loss_rand_sum.backward()
                self.optimizer.step()

                if det.sum().item() > 0:
                    rand_pos_loss = (det * loss_rand).sum() / (det.sum())
                    loss_r_pos = 0.95 * loss_r_pos + 0.05 * rand_pos_loss.item()
                if (1-det).sum().item() > 0:
                    rand_neg_loss = ((1-det) * loss_rand).sum() / ((1-det).sum())
                    loss_r_neg = 0.95 * loss_r_neg + 0.05 * rand_neg_loss.item()

                description = ("Loss: %0.4f, lr: %0.6f, rand pos: %.1f, rand neg: %.1f" %
                                            (loss_avg,
                                            self.optimizer.param_groups[0]['lr'],
                                            loss_r_pos,
                                            loss_r_neg))
            else:
                description = ("Loss: %0.4f, lr: %0.6f" %
                                            (loss_avg,
                                            self.optimizer.param_groups[0]['lr']))
            update_progress(i, batch_count, start, text=description)
            
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.save_embedding(
            self.data.id2word, self.output_file_name, self.use_cuda)
