import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import seaborn as scs
import pylab as plot
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hyper.adult import read_data
from hyper.causal_utils import (
    get_query_output,
)

df=read_data()

backdoor={'Age':[],'sex':[],'country':[],'marital':['country','Age','sex'],'edu':['Age','sex','country'],
              'class':['Age','sex','country'],'occupation':['Age','sex','country'],
              'hours':['Age','sex','country']}

X=df

scores={}
raw_scores={}
for col in ['marital','occupation','edu','class']:
    values= list(set(df[col].values))
    scores[col]=[]
    for v in values:
        scores[col].append(get_query_output(df,'count','',[],[],['target'],[1],[col],[v],['*'],'',{}, backdoor))#,{0:[1,2]}))
        raw_scores[(col,v)]=scores[col][-1]


hyper=[]
hypermax=[]
for col in ['marital','occupation','edu','class']:
    hyper.append(min(scores[col]))
    hypermax.append(max(scores[col]))
print(hyper,hypermax)

hypername=['Marital', 'Occupation', 'Education','Class']

hypermaxname=['Marital', 'Occupation', 'Education','Class']


num_feat=4

fsize=20
params = {'legend.fontsize': fsize/1.2,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : fsize}
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = hypername
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
plt.figure(figsize=(6, 10)) # in inches!

fig, ax = plt.subplots()
rects1 = ax.barh(x - 0.15, hyper, width, label='Minimum', color='lightcoral', edgecolor='black', hatch="//")
rects2 = ax.barh(x, hypermax, width, label='Maximum', color='gainsboro', edgecolor='black', hatch="\\\\")
#rects3 = ax.barh(x + 0.15, baseline_all, width, label='Hyper-NB', color='forestgreen', hatch='|')
#rects4 = ax.barh(x + 0.3, baseline, width, label='Indep', color='darkviolet', hatch='+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Query Output', fontsize=fsize, labelpad=fsize/2)
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=fsize)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xticks(np.arange(0, 2, .5))
ax.legend(loc=(0.52,0))
ax.invert_yaxis()
ax.margins(0.1,0.05)
plt.xlim([0, 1])
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('../freshRuns/8b.pdf')