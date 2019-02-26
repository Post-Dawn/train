import findspark
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
print(rdd.collect())