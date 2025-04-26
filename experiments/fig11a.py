import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import pylab as plot
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
from hyper.germansyn import get_data
from hyper.causal_utils import (
    get_query_output,
)

backdoor={'S':[],'A':[],'St':['S','A'],'sav':['S','A'],'hous':['S','A'],'Cred':['S','A']}


scores={}
times={}
sampled_times={}
for size in [10000,100000,200000,400000,600000,800000,1000000]:
    (orig_df,df_U)=get_data(size,0)


    print (size)
    times[size]=[]
    sampled_times[size]=[]
    
    for col in ['St','hous','sav','Cred','S']:
    

        start=time.time()
        score=get_query_output(orig_df,'count','',[],[],['Y'],[1],[col],[1],['*'],'',{}, backdoor)
        end=time.time()
        times[size].append(end-start)

        start=time.time()
        df=orig_df.sample(n=10000,random_state=0)
        score=get_query_output(df,'count','',[],[],['Y'],[1],[col],[1],['*'],'',{}, backdoor)
        end=time.time()
        sampled_times[size].append(end-start)
    
print (times)
print (sampled_times)


backdoor={'S':[],'A':[],'St':[],'sav':[],'hous':[],'Cred':[]}

indep_times={}
for size in [10000,100000,200000,400000,600000,800000,1000000]:
    (orig_df,df_U)=get_data(size,0)


   
    feat=list(df.columns)
    indep_times[size]=[]
    
    for col in ['St','hous','sav','Cred','S']:
    
        start=time.time()
        score=get_query_output(orig_df,'count','',[],[],['Y'],[1],[col],[1],['*'],'',{}, backdoor)
        end=time.time()
        indep_times[size].append(end-start)


print (indep_times)


indep=[]
our=[]
oursampled=[]
for size in [10000,100000,200000,400000,600000,800000,1000000]:
    indep.append(np.mean(indep_times[size]))
    our.append(np.mean(times[size]))
    oursampled.append(np.mean(sampled_times[size]))


x=[0.010,0.1,0.2,0.4,0.6,0.8,1.0]

import pylab as plot
params = {'legend.fontsize': 65,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : 65}
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = snname=[0.010,0.1,0.2,0.4,0.6,0.8,1.0]

x = np.arange(len(labels))  # the label locations

width = 0.25  # the width of the bars

plt.figure(figsize=(20, 20)) # in inches!

#fig, ax = plt.subplots()
#rects1 = ax.barh(x - width, trsn, width,xerr=trsnvar, label='Ground Truth', color='coral', edgecolor='black', hatch="/",error_kw=dict(elinewidth=5, ecolor='black'))
#rects2 = ax.barh(x, sn, width, xerr=snvar,label='RAVEN', color='forestgreen', edgecolor='black', hatch="||",error_kw=dict(elinewidth=5, ecolor='black'))
#rects3 = ax.barh(x + width, allScores['sn'], width, label='NeSuf', color='royalblue')

y = np.array(our)
x = np.array([0.010,0.1,0.2,0.4,0.6,0.8,1.0])

#error = np.array([0.02,0.003,0.0023,0.0001,0.001])#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)


y1 = np.array(indep)
y2=np.array(oursampled)
#error1 = np.array(snvar)#np.random.normal(0.1, 0.02, size=y.shape)
#y += np.random.normal(0, 0.1, size=y.shape)
#print(error1)

plt.plot(x, y, 'v-',label='HypeR',linewidth=5,color='black',markersize=40)
plt.plot(x, y2, 's-',label='HypeR-sampled',linewidth=5,color='pink',markersize=40)
plt.plot(x, y1, 's-',label='Indep',linewidth=5,color='forestgreen',markersize=40)

plt.fill_between(x, y, y, alpha=0.3)
plt.fill_between(x, y1, y1, alpha=0.15,color='forestgreen')
#plt.show()
#plt.xticks([0, 1000,50000, 100000], ['', '1K','50K', '100K'])
#plot.ylim([0.4,0.7])
plt.legend()
plt.xlabel('Dataset Size (in millions)',labelpad=30)
plt.ylabel('Time (in seconds)',labelpad=-11)
plt.savefig('../freshRuns/11a.pdf', bbox_inches='tight')