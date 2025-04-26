import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import pylab as plot
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
from hyper.germansyn import get_data, causal_effect
from hyper.causal_utils import (
    get_query_output,
)


(df,df_U)= get_data(1000000,0)
feat=list(df.columns)

backdoor={'S':[],'A':[],'St':['S','A'],'sav':['S','A'],'hous':['S','A'],'Cred':['S','A']}

sampled_df=df.sample(n=100000,random_state=0)
scores={}
sampled_scores={}
for col in ['St','sav','hous','Cred']:
    values= list(set(df[col].values))
    scores[col]=[]
    #for v in values:
    scores[col]=get_query_output(df,'count','',[],[],['Y'],[1],[col],[1],['*'],'',{},backdoor)
    sampled_scores[col]=get_query_output(sampled_df,'count','',[],[],['Y'],[1],[col],[1],['*'],'',{}, backdoor)


indep_scores={}
gt_score={}
for col in ['St','sav','hous','Cred']:
    vals=df[df[col]==1]['Y'].value_counts()
    indep_scores[col]=vals[1]*1.0/(vals[0]+vals[1])
    
    modified= causal_effect({col:1},df,df_U)
    after_intervention=modified['Y'].value_counts()
    gt_score[col]=after_intervention[1]*1.0/(after_intervention[0]+after_intervention[1])


backdoor={'S':[],'A':[],'St':['S','A','sav','hous','Cred'],'sav':['S','A','St','hous','Cred'],'hous':['S','A','sav','St','Cred'],'Cred':['S','A','sav','hous','St']}

sampled_scores_nb={}
for col in ['St','sav','hous','Cred']:
    values= list(set(df[col].values))
    scores[col]=[]
    #for v in values:
    
    sampled_scores_nb[col]=get_query_output(sampled_df,'count','',[],[],['Y'],[1],[col],[1],['*'],'',{}, backdoor)


hyper=[]
gt=[]
hypersampled=[]
baseline=[]
baseline_all=[]
for col in ['St','sav','hous','Cred']:
    hyper.append(scores[col])
    gt.append(gt_score[col])
    hypersampled.append(sampled_scores[col])
    baseline.append(indep_scores[col])
    baseline_all.append(sampled_scores_nb[col])

import seaborn as scs
hypername=['Status', 'Savings', 'Housing','Credit Amount']

gtname=['Status', 'Savings', 'Housing','Credit\n Amount']

#gt=[0.99,0.6298,0.507,0.5007,0.02]
#gtname=[ 'Age',  'Saving','Status','Sex', 'Housing']

hypernamesampled=['Status', 'Savings', 'Housing','Credit Amount']

baselinename=['Status', 'Savings', 'Housing','Credit Amount']


baselinename_all=['Status', 'Savings', 'Housing','Credit Amount']



num_feat=4
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pylab as plot
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
rects1 = ax.barh(x - 0.30, gt, width, label='Ground Truth', color='lightcoral', edgecolor='black', hatch="//")
rects5 = ax.barh(x - 0.15, hypersampled, width, label='Hyper-sampled', color='skyblue',edgecolor='black', hatch='|')
#rects2 = ax.barh(x, hyper, width, label='Hyper', color='gainsboro', edgecolor='black',hatch="\\\\")
rects3 = ax.barh(x, baseline_all, width, label='Hyper-NB', color='forestgreen',edgecolor='black', hatch='|')
rects4 = ax.barh(x + 0.15, baseline, width, label='Indep', color='darkviolet', edgecolor='black',hatch='+')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Query Output', fontsize=fsize, labelpad=fsize/2)
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=fsize)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xticks(np.arange(0, 2, .5))
ax.legend(loc=(0.49,0.01))
ax.invert_yaxis()
ax.margins(0.1,0.05)
plt.xlim([0, 1.3])
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('../freshRuns/10a.pdf')


