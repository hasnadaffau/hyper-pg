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
from hyper.germansyn import get_data_stud, causal_effect
from hyper.causal_utils import (
    get_query_output,
)

(df,df_U)=get_data_stud(1000000,0)
for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    df[col+'_disc']=pd.cut(df[col],
           bins=[df[col].describe()['min']-1, df[col].describe()['50%'], df[col].describe()['max']], 
           labels=[0, 1])
feat=list(df.columns)
backdoor={'S':[],'A':[],'Country':[],'hands_raised':['S','A','Country'],'Attendance':['S','A','Country'],'discussion':['S','A','Country','Attendance'],'assignment':['S','A','Country'],
         'announcement':['S','A','Country']}

sampled_df=df.sample(n=10000,random_state=0)
sampled_scores=[]

for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    sampled_scores.append(get_query_output(sampled_df,'avg','grade',[],[],[],[],[col],[1],['*'],'',{}, backdoor))



backdoor={'S':[],'A':[],'Country':[],'hands_raised':['S','A','Country','assignment_disc','Attendance_disc','announcement_disc','discussion_disc'],'Attendance':['S','A','Country','assignment_disc','announcement_disc','hands_raised_disc','discussion_disc'],'discussion':['S','A','Country','Attendance_disc','assignment_disc','Attendance_disc','announcement_disc','hands_raised_disc'],'assignment':['S','A','Country','Attendance_disc','announcement_disc','hands_raised_disc','discussion_disc'],
         'announcement':['S','A','Country','assignment_disc','Attendance_disc','hands_raised_disc','discussion_disc']}

sampled_scores_nb=[]

for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    print (col)
    sampled_scores_nb.append(get_query_output(sampled_df,'avg','grade',[],[],[],[],[col],[1],['*'],'',{}, backdoor))


gt=[]
for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    intervened=causal_effect({col:1},df,df_U)
    gt.append(intervened['grade'].describe()['mean'])

backdoor={'S':[],'A':[],'Country':[],'hands_raised':[],'Attendance':[],'discussion':[],'assignment':[],
         'announcement':[]}

indep=[]

for col in ['assignment','Attendance','announcement','hands_raised','discussion']:
    indep.append(get_query_output(sampled_df,'avg','grade',[],[],[],[],[col],[1],['*'],'',{}, backdoor))

hyper=sampled_scores

hypername=['Assignment', 'Attendance', 'Announcement','Hand Raised', 'Discussion']

gt=gt
gtname=['Assignment', 'Attendance', 'Announcement','Hand Raised', 'Discussion']

#gt=[0.99,0.6298,0.507,0.5007,0.02]
#gtname=[ 'Age',  'Saving','Status','Sex', 'Housing']


baseline=indep
baselinename=['Assignment', 'Attendance', 'Announcement','Hand Raised', 'Discussion']


hyper_all=sampled_scores_nb


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
rects1 = ax.barh(x - 0.15, gt, width, label='Ground Truth', color='lightcoral', edgecolor='black', hatch="//")
rects2 = ax.barh(x, hyper, width, label='Hyper-sampled', color='gainsboro', edgecolor='black', hatch="\\\\")
rects3 = ax.barh(x + 0.15, hyper_all, width, label='Hyper-NB', color='forestgreen', hatch='|')
rects4 = ax.barh(x + 0.3, baseline, width, label='Indep', color='darkviolet', hatch='+')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Query Output', fontsize=fsize, labelpad=fsize/2)
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=fsize)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xticks(np.arange(0, 150, 50))
ax.legend(loc=(0.384,0))
ax.invert_yaxis()
plt.xlim([0, 120])
ax.margins(0.1,0.05)
matplotlib.rcParams['hatch.linewidth'] = 0.2
figure = plt.gcf() # get current figure
figure.set_size_inches(7,7.5)
fig.tight_layout()
plt.savefig('../freshRuns/10b.pdf')

