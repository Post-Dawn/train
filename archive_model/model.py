
import torch
import torch.nn as nn
import torch.nn.functional as F

class all2vec(nn.Module):

    def __init__(self, n_vertices, n_dimension):
        super(all2vec, self).__init__()
        self.embedding = nn.Embedding(n_vertices, n_dimension)
        self.embedding_context = nn.Embedding(n_vertices, n_dimension)
        self.n_dimension = n_dimension


    def first_proximity(self, pos_v, pos_u, neg_u, weights):
        
        emb_v = self.embedding(pos_v)
        emb_u = self.embedding(pos_u)
        emb_neg_u = self.embedding_context(neg_u)

        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)

        neg_score = torch.bmm(emb_neg_u, emb_v.unsqueeze(2)).squeeze(2)
        neg_score = F.logsigmoid(-neg_score)
        neg_score = torch.sum(neg_score, 1)

        score = -F.logsigmoid(score) - neg_score
        score = torch.mul(score, weights).squeeze()
        #score = torch.sum(score)
        return score


    def second_proximity(self, pos_v, pos_u, neg_u, weights):
        emb_v = self.embedding(pos_v)
        emb_u = self.embedding_context(pos_u)
        emb_neg_u = self.embedding_context(neg_u)

        neg_score = torch.bmm(emb_neg_u, emb_v.unsqueeze(2)).squeeze(2)
        neg_score = F.logsigmoid(-neg_score)
        neg_score = torch.sum(neg_score, 1)

        pos_score = torch.mul(emb_v, emb_u).sum(1)
        
        score = -F.logsigmoid(pos_score) - neg_score
        score = torch.mul(weights, score)

        return score


    def forward(self, pos, neg):

        pos_v = pos[:,0].long()
        pos_u = pos[:,1].long()
        weights = pos[:, 2]
        return self._forward(pos_v, pos_u, neg, weights)

    def _forward(self, pos_v, pos_u, neg_u, weights):
        score_1 = self.first_proximity(pos_v, pos_u, neg_u, weights)
        score_2 = self.second_proximity(pos_v, pos_u, neg_u, weights)
        return score_1, score_2

