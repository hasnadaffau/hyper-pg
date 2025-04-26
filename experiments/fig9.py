import os
import sys
import time
import copy
import json
import numpy as np
import pandas as pd
import matplotlib
import seaborn as scs
import pylab as plot
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hyper.causal_utils import (
    get_logistic_param,
    optimization,
    get_combination,
    get_C_set,
    get_data
    
)

config_file=open('../config.json','r')
configs=json.load(config_file)
print (configs)

max_time=int(configs['max_runtime'])*60*60

scores={}
times={}
total_time=0
opt_times={}
opt_scores={}
for num_bins in [1,2,4,6,8,10]:
    print (num_bins)
    i=0
    lst=[]
    while i<num_bins:
        lst.append(i+1)
        i+=1
    
    (df,df_U,bins)=get_data(20000,0,num_bins)

    print (df)
    A=['St_orig','sav_orig','hous_orig','Cred_orig']#,'housing','employment']
    aval=[1,0,0]

    klst=['A','S']
    kval=[]

    target='Y'
    start=time.time()
    conditionals=copy.deepcopy(A)
    conditionals.extend(klst)
    #print (conditionals)
    (beta_lst,[[beta0]])=get_logistic_param(df,conditionals,[target],[1])
    print(beta_lst,beta0)


    Adomain=[lst,lst,lst,lst]#,[0,1],[0,1,2,3,4]]#List of list where each list is domain of each variable in A

    scores[num_bins],gt_score=optimization(df,A,aval,Adomain,klst,kval,0.8,beta_lst,beta0, bins)
    end=time.time()
    times[num_bins]=end-start
    if total_time >= max_time:
        opt_times[num_bins]=opt_times[num_bins-2]*opt_times[2]
        opt_scores[num_bins]=opt_scores[num_bins-2]
        continue
    start=time.time()
    domain_lst=get_combination(Adomain,[])
    maxval=0
    for lst in domain_lst:
        print(lst)
        St_bins=bins['St']
        sav_bins=bins['sav']
        hous_bins=bins['hous']
        cred_bins=bins['Cred']
        (beta_lst,[[beta0]])=get_logistic_param(df,conditionals,[target],[1])
        print (conditionals,beta_lst)
        stval=beta_lst[0]*((St_bins[lst[0]-1]+St_bins[lst[0]])/2)
        savval=beta_lst[1]*((sav_bins[lst[0]-1]+sav_bins[lst[0]])/2)
        housval=beta_lst[2]*((hous_bins[lst[0]-1]+hous_bins[lst[0]])/2)
        credval=beta_lst[3]*((cred_bins[lst[0]-1]+cred_bins[lst[0]])/2)
    
        backdoorvals=get_C_set(df,klst)
    
        betalst_backdoor=beta_lst[len(A):]
        sum_backdoor=0
        for lst in backdoorvals:
            iter=0
            sampled_df=copy.deepcopy(df)
            tmpsum=0
            while iter<len(lst):
                sampled_df=sampled_df[sampled_df[klst[iter]]==lst[iter]]
                tmpsum+=betalst_backdoor[iter]*lst[iter]
                iter+=1
            sum_backdoor+= tmpsum*(sampled_df.shape[0]*1.0/df.shape[0])

            print(sampled_df.shape[0])
        if sum_backdoor+stval+savval+housval+credval> maxval:
            maxval=sum_backdoor+stval+savval+housval+credval
        print (sum_backdoor+stval+savval+housval)
    print ("sum backdoor",sum_backdoor,stval,savval,housval) 
    end=time.time()
    opt_times[num_bins]=end-start
    opt_scores[num_bins]=(maxval+beta0)*1.0/gt_score
    print (end-start)
    total_time+=opt_times[num_bins]
#(A,aval,Adomain,klst,kval,alpha,betalst,beta0):

print (scores)
print (opt_scores)
print(times)
print(opt_times)

hyper=[]
baseline=[]
for num_bins in [1,2,4,6,8,10]:
    hyper.append(scores[num_bins])
    baseline.append(opt_scores[num_bins])

hypername=[1,2,4,6,8,10]

gtname=[1,2,4,6,8,10]

#gt=[0.99,0.6298,0.507,0.5007,0.02]
#gtname=[ 'Age',  'Saving','Status','Sex', 'Housing']


#baseline=[0.640,0.76,0.91,0.92, 0.95, 0.98]
baselinename=[1,2,4,6,8,10]


#baseline_all=[0.64,0.44, 0.57,0.43]
baselinename_all=[1,2,4,6,8,10]


num_feat=4

fsize=20
params = {'legend.fontsize': fsize/1.2,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : fsize}
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = gtname
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
plt.figure(figsize=(6, 10)) # in inches!

fig, ax = plt.subplots()
#rects1 = ax.bar( x - 0.15,gt, width, label='Ground Truth', color='lightcoral', edgecolor='black', hatch="//")
rects2 = ax.bar(x, hyper, width, label='Hyper-sampled', color='gainsboro', edgecolor='black', hatch="\\\\")
#rects3 = ax.bar(x + 0.15, baseline_all, width, label='Hyper-NB', color='forestgreen', hatch='|')
rects4 = ax.bar(x + 0.15, baseline, width, label='Opt-discrete', color='darkviolet', hatch='+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Solution Quality', fontsize=fsize, labelpad=fsize/2)
ax.set_xlabel('Number of buckets', fontsize=fsize, labelpad=fsize/2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=fsize)
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.yticks(np.arange(0, 2, .5))
ax.legend(loc=(0.5,0.85))
ax.margins(0.1,0.05)
plt.ylim([0, 1.3])
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('../freshRuns/9a.pdf')



hyper=[]
baseline=[]

for num_bins in [1,2,4,6,8,10]:
    hyper.append(times[num_bins])
    baseline.append(opt_times[num_bins])


hypername=[1,2,4,6,8,10]

gtname=[1,2,4,6,8,10]

#gt=[0.99,0.6298,0.507,0.5007,0.02]
#gtname=[ 'Age',  'Saving','Status','Sex', 'Housing']


baselinename=[1,2,4,6,8,10]

baselinename_all=[1,2,4,6,8,10]

fsize=20
params = {'legend.fontsize': fsize/1.2,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : fsize}
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = gtname
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
plt.figure(figsize=(6, 10)) # in inches!

fig, ax = plt.subplots()
#rects1 = ax.bar( x - 0.15,gt, width, label='Ground Truth', color='lightcoral', edgecolor='black', hatch="//")
rects2 = ax.bar(x, hyper, width, label='Hyper-sampled', color='gainsboro', edgecolor='black', hatch="\\\\")
#rects3 = ax.bar(x + 0.15, baseline_all, width, label='Hyper-NB', color='forestgreen', hatch='|')
rects4 = ax.bar(x + 0.15, baseline, width, label='Opt-discrete', color='darkviolet', hatch='+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Running Time (seconds)', fontsize=fsize, labelpad=fsize/2)
ax.set_xlabel('Number of buckets', fontsize=fsize, labelpad=fsize/2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=fsize)
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.yticks(np.arange(0, 3.5, .5))
ax.legend(loc=(0.25,0.85))
ax.margins(0.1,0.05)
plt.yscale("log")

plt.ylim([0, 35000])
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('../freshRuns/9b.pdf')