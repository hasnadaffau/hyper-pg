import pandas as pd
import numpy as np

def random_logit(x):
    z = 1./(1+np.exp(-x))
    #print(z) 
    s = np.random.binomial(n=1, p=z)

    return s
def roundlst(x):
    l=[]
    for v in x:
        if v>0.5:
            l.append(1)
        else:
            l.append(0)
    return l
def roundl(x,th):
    l=[]
    for v in x:
        if v>th:
            l.append(1)
        else:
            l.append(0)
    return l

def roundl4(x,th):
    l=[]
    for v in x:
        if v<=th:
            l.append(0)
        elif v<=2*th:
            l.append(1)
        elif v<=3*th:
            l.append(2)
        else:# v<=4*th:
            l.append(3)
    return l

def bucketize(lst,num_buckets):
    maxval=max(lst)
    minval=min(lst)
    binlst=[minval-0.001]
    i=0
    labels=[]
    size=(maxval-minval)/num_buckets
    while i<num_buckets:
        binlst.append(binlst[-1]+size)#0.001)
        labels.append(i)
        i+=1
    binlst[-1]+=0.001
    return (pd.cut(x=lst, bins=binlst,labels=labels),binlst)


