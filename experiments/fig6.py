import os
import sys
import time
import json
import numpy as np
import pandas as pd
import statistics
import matplotlib
import pylab as plot
from matplotlib import pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
from hyper.germansyn import get_data
from hyper.causal_utils import (
    get_query_output,
)

# def load_config():
#     """Load config.json from project root"""
#     dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     with open(os.path.join(dir_path, 'config.json')) as f:
#         return json.load(f)
    
# #Running ExperimentsS
# configs = load_config()
backdoor={'S':[],'A':[],'St':['S','A'],'sav':['S','A'],'hous':['S','A'],'Cred':['S','A']}

df, df_U = get_data(10000,0)
orig_df, df_U = get_data(1000000,0)
print("Running on 1 milion samples...")
orig_score=get_query_output(orig_df,'count','',[],[],['Y'],[1],['St'],[1],['*'],'',{}, backdoor)
print ("completed 1million")

scores={}
times={}
for size in [1000,10000,25000,50000,100000,200000]:
    scores[size]=[]
    times[size]=[]
    for seed in [0,1,2,3,4,5,6,7,8,9,10]:
        print (seed,size)
        df=orig_df.sample(n=size,random_state=seed)
        feat=list(df.columns)
        start=time.time()
        score=get_query_output(df,'count','',[],[],['Y'],[1],['St'],[1],['*'],'',{}, backdoor)
        end=time.time()
        times[size].append (end-start)
        scores[size].append(score)
print (scores)
print (times)

runtime=[]
for size in [10000,100000,200000,400000,600000,800000,1000000]:
    if size in times.keys():
        runtime.append(np.mean(times[size]))
        continue
    df=orig_df.sample(n=size,random_state=0)
    feat=list(df.columns)
    start=time.time()
    score=get_query_output(df,'count','',[],[],['Y'],[1],['St'],[1],['*'],'',{}, backdoor)
    end=time.time()
    runtime.append (end-start)

print (runtime)

average=[]
dev=[]
gt=[]
for size in scores.keys():
    average.append (np.mean(scores[size]))
    dev.append (statistics.stdev(scores[size]))
    gt.append(orig_score)
print (average)
print(dev)


#Produce Fig6a
fsize=20
params = {'legend.fontsize': fsize,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : fsize}
matplotlib.rc('font', **font)

labels = [1000,10000,25000,50000,100000,200000]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

plt.figure(figsize=(6, 5)) # in inches!
y = np.array([0.6130208937041055,0.6130208937041055,0.6130208937041055,0.6130208937041055,0.6130208937041055,0.6130208937041055])
x = np.array([1000,10000,25000,50000,100000,200000])
#error = np.array([0.02,0.003,0.0023,0.0001,0.001])#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)


y1 = np.array(average)
error1 = np.array(dev)#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)
print(error1)

plt.xticks(fontsize= fsize/1.2)
plt.plot(x, y, 'v-',label='Ground Truth',color='salmon',markersize=18)
plt.plot(x, y1, 'x-',label='Hyper-sampled',color='forestgreen',markersize=18)
plt.fill_between(x, y, y, alpha=0.3)
plt.fill_between(x, y1-error1, y1+error1, alpha=0.15,color='forestgreen')
plt.xticks([0, 1000,50000, 100000,200000], ['', '1K','50K', '100K','200K'])
plot.ylim([0.55,0.85])
plt.legend()
plt.xlabel('Sample Size',labelpad=5, fontsize=fsize/1.1)
plt.savefig('../freshRuns/6a.pdf', bbox_inches='tight')


#produce Fig6b
x=[0.010,0.1,0.2,0.4,0.6,0.8,1.0]
our=[390,390,390,390,390,390,390]
oursampled=[5.7,44.5,88.2,178.4,240,320,390]
indep=[1.1,8.9,19.2,36,55,74,95]
#ourNB=[22.4,222.5,80.2,164.4,232,310,390]
    
import pylab as plot
params = {'legend.fontsize': 65,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : 65}
matplotlib.rc('font', **font)

labels = snname=[0.010,0.1,0.2,0.4,0.6,0.8,1.0]

x = np.arange(len(labels))  

width = 0.25  

plt.figure(figsize=(20, 20))

y = np.array([runtime[-1],runtime[-1],runtime[-1],runtime[-1],runtime[-1],runtime[-1],runtime[-1]])
x = np.array([0.010,0.1,0.2,0.4,0.6,0.8,1.0])
y2=np.array(runtime)


plt.plot(x, y, 'v-',label='HypeR',linewidth=5,color='black',markersize=40)
plt.plot(x, y2, 's-',label='HypeR-sampled',linewidth=5,color='pink',markersize=40)
plt.fill_between(x, y, y, alpha=0.3)
plt.legend()
plt.xlabel('Sample Size (in millions)',labelpad=30)
plt.ylabel('Time (in seconds)',labelpad=-11)
plt.savefig('../freshRuns/6b.pdf', bbox_inches='tight')

