'''import findspark
findspark.init()
from pyspark import SparkContext

def double(x):
    ret = []
    for i in x:
        now=[]
        for j in i:
            now.append(j*2)
        ret.append(tuple(now))
    ret = tuple(ret)
    fret = []
    fret.append(ret)
    return fret

sc=SparkContext()
now=[((1.2,2.0,3,3,3,3,3,3,3,3),(2.3,3.0),(1.2,3.3))]
rdd=sc.parallelize(now)
rdd=rdd.map(double)
print(rdd.collect())'''
'''from model import SkipGramModel
import torch
a = [1,2,3]
print(torch.ones(3))'''
#skip_gram_model = SkipGramModel(2, 3)
#skip_gram_model.u_embeddings.weight = a
#skip_gram_model.v_embeddings.weight.data=torch.Tensor(a)
#print(skip_gram_model.v_embeddings.weight)
import torch
import torch.nn as nn
embedding = nn.Embedding(10, 3)
input = torch.FloatTensor([[1,2,4,5],[4,3,2,9]])
print(input)