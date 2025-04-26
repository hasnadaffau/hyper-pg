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
from hyper.germansyn import get_data_vary_var
from hyper.causal_utils import (
    get_query_output,
)

scores=[]
times=[]
backdoor={'S':[],'A':[],'Country':[],'hands_raised':['S','A','Country'],'Attendance':['S','A','Country'],'discussion':['S','A','Country','Attendance'],'assignment':['S','A','Country'],
         'announcement':['S','A','Country']}
    
for num_var in [0,3,5,8,10]:
    (new_df,newdf_U)=get_data_vary_var(10000,0,num_var)
    i=0
    prelst=[]
    prevallst=[]
    while i<num_var:
        #backdoor['hands_raised'].append("var"+str(i))
        prelst.append("var"+str(i))
        prevallst.append(0)
        i+=1
    start=time.time()
    scores.append (get_query_output(new_df,'avg','grade',prelst,prevallst,[],[],['hands_raised'],[1],['*'],'',{}, backdoor))
    end=time.time()
    times.append(end-start)


indep_times=[]
backdoor={'S':[],'A':[],'Country':[],'hands_raised':[],'Attendance':[],'discussion':[],'assignment':[],
         'announcement':[]}
    
for num_var in [0,3,5,8,10]:
    (new_df,newdf_U)=get_data_vary_var(10000,0,num_var)
    i=0
    prelst=[]
    prevallst=[]
    while i<num_var:
        #backdoor['hands_raised'].append("var"+str(i))
        prelst.append("var"+str(i))
        prevallst.append(0)
        i+=1
    start=time.time()
    scores.append (get_query_output(new_df,'avg','grade',prelst,prevallst,[],[],['hands_raised'],[1],['*'],'',{}, backdoor))
    end=time.time()
    indep_times.append(end-start)

x=[0,3,5,8,10]
our=times
opt=indep_times
    
import pylab as plot
params = {'legend.fontsize': 65,
          'legend.handlelength': 2}
plot.rcParams.update(params)

font = {'family' : "sans serif", 'size'   : 65}
# matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('font', **font)

labels = snname=['Sex','Age','Status','Saving','Housing']

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

plt.figure(figsize=(20, 20)) # in inches!
y = np.array(our)
x = np.array([0,3,5,8,10])

y1 = np.array(opt)

plt.plot(x, y, 'v-',label='HypeR-sampled',linewidth=5,color='black',markersize=40)
plt.plot(x, y1, 's-',label='Indep',linewidth=5,color='forestgreen',markersize=40)
plt.fill_between(x, y, y, alpha=0.3)
plt.fill_between(x, y1, y1, alpha=0.15,color='forestgreen')
#plt.show()
#plt.xticks([0, 1000,50000, 100000], ['', '1K','50K', '100K'])
#plot.ylim([0.4,0.7])
plt.legend()
plt.xlabel('Attributes',labelpad=30)
plt.ylabel('Time (in seconds)',labelpad=0)
plt.savefig('../freshRuns/12a.pdf')