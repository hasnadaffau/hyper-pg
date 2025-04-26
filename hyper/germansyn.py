import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# from core.helper import random_logit, roundlst, roundl, roundl4, bucketize
import hyper.helper as helper

def get_data(N,seed):
    #N=10000
    np.random.seed(seed)
    S=np.random.binomial(n=1, p=0.5,size=N)
    A=np.random.binomial(n=2, p=0.5,size=N)
    noise1=np.random.binomial(n=2, p=0.05,size=N)
    noise2=np.random.binomial(n=2, p=0.05,size=N)
    noise3=np.random.binomial(n=2, p=0.05,size=N)
    noise4=np.random.binomial(n=2, p=0.05,size=N)
    St=np.array(helper.roundl((2*S+A+ noise1 )/2,1))
    
    Cred=np.array(helper.roundl((0.5*S+1.5*A+ noise1 )/2,1))
    
    sav=np.array(helper.roundl((S+A+noise2)/3,0.5))
            
    hous=np.array(helper.roundl((S+noise3)/2,0.5))
    
    #X1 = (2*S+A+np.random.normal(loc=0.0, scale=0.2, size=N))#(np.random.normal(loc=100.0, scale=1.16, size=N)) # N(0,1)
    #X2 =(3*A+2*S+np.random.normal(loc=0.0, scale=0.2, size=N))#random_logit(100*X1)#+np.random.normal(loc=0.0, scale=0.16, size=N)
    #Y = random_logit((A+3*St+2*sav+hous)/50)
    Y = helper.roundlst((St/3+sav/3+hous/3+noise4))
    df=pd.DataFrame(S,columns=['S'])
    df_U=pd.DataFrame(S,columns=['S'])
    df_U['noise1']=noise1
    df_U['noise2']=noise2
    df_U['noise3']=noise3
    df_U['noise4']=noise4
    df['A']=A
    df['S']=S
    df['St']=St
    df['Cred']=Cred
    df['sav']=sav
    df['hous']=hous
    df['Y']=Y
    
    
    return (df,df_U)

def get_data_stud(N,seed):
    #N=10000
    #np.random.seed(seed)
    
    S=np.random.binomial(n=1, p=0.5,size=N)
    A=np.random.binomial(n=2, p=0.5,size=N)
    Country=np.random.binomial(n=2, p=0.5,size=N)
    
    
    Attendance_U=np.random.binomial(n=4, p=0.5,size=N)
    Attendance=Attendance_U + S + A + Country
    
    hands_U=np.random.normal(10, 2, size=N)
    hands_raised=hands_U+ 2*S+ 3*A + Country
    
    discussion_U=np.random.normal(10, 2, size=N)
    discussion=discussion_U+ 2*Attendance + 2*S+ 3*A + Country
    
    assignment_U=np.random.normal(10, 2, size=N)
    assignment=assignment_U+ 2*S+ 3*A + Country -3*Attendance
    
    announcement_U=np.random.normal(10, 2, size=N)
    announcement=announcement_U+ 2*S+ 3*A + Country
    
    
    grade_U=np.random.normal(10, 2, size=N)
    grade=grade_U+ hands_raised + 2*discussion + assignment + announcement
    
    df=pd.DataFrame(S,columns=['S'])
    df_U=pd.DataFrame(S,columns=['S'])
    
    df['A']=A
    df['S']=S
    df['Country']=Country
    df['hands_raised']=hands_raised
    df['Attendance']=Attendance
    df['discussion']=discussion
    df['assignment']=assignment
    df['announcement']=announcement
    df['grade']=grade
    
    
    df_U['A']=A
    df_U['S']=S
    df_U['Country']=Country
    df_U['hands_raised']=hands_U
    df_U['Attendance']=Attendance_U
    df_U['discussion']=discussion_U
    df_U['assignment']=assignment_U
    df_U['announcement']=announcement_U
    df_U['grade']=grade_U
    
    
    return (df,df_U)

def get_data_vary_var(N,seed,num_var):
    #N=10000
    np.random.seed(seed)
    
    S=np.random.binomial(n=1, p=0.5,size=N)
    A=np.random.binomial(n=2, p=0.5,size=N)
    Country=np.random.binomial(n=2, p=0.5,size=N)
    
    
    Attendance_U=np.random.binomial(n=4, p=0.5,size=N)
    Attendance=Attendance_U + S + A + Country
    
    hands_U=np.random.normal(10, 2, size=N)
    hands_raised=hands_U+ 2*S+ 3*A + Country
    
    discussion_U=np.random.normal(10, 2, size=N)
    discussion=discussion_U+ 2*Attendance + 2*S+ 3*A + Country
    
    assignment_U=np.random.normal(10, 2, size=N)
    assignment=assignment_U+ 2*S+ 3*A + Country -3*Attendance
    
    announcement_U=np.random.normal(10, 2, size=N)
    announcement=announcement_U+ 2*S+ 3*A + Country
    
    grade_U=np.random.normal(10, 2, size=N)
    grade=grade_U+ hands_raised + 2*discussion + assignment + announcement
    
    
      
    df=pd.DataFrame(S,columns=['S'])
    df_U=pd.DataFrame(S,columns=['S'])
    
    df['A']=A
    df['S']=S
    df['Country']=Country
    df['hands_raised']=hands_raised
    df['Attendance']=Attendance
    df['discussion']=discussion
    df['assignment']=assignment
    df['announcement']=announcement
    df['grade']=grade
    
    i=0
    while i<num_var:
        df["var"+str(i)] = helper.roundlst(np.random.normal(0, 2, size=N) + np.random.rand()*Attendance)
        i+=1
    
    
    df_U['hands']=hands_U
    df_U['Attendance']=Attendance_U
    df_U['discussion']=discussion_U
    df_U['assignment']=assignment_U
    df_U['announcement']=announcement_U
    df_U['grade']=grade_U
    
    
    return (df,df_U)


def causal_effect(dolst,df,df_U):

    N=df.shape[0]
    if 'S' in dolst.keys():
        S=np.array([dolst['S']]*df.shape[0])#Construct a list of same size as df
    else:
        S=df['S']
        
        
    if 'A' in dolst.keys():
        A=np.array([dolst['A']]*df.shape[0])#Construct a list of same size as df
    else:
        A=df['A']


    noise1=df_U['noise1']
    noise2=df_U['noise2']
    noise3=df_U['noise3']
    noise4=df_U['noise4']
    if 'St' in dolst.keys():
        St= np.array([dolst['St']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys():
        St= np.array(helper.roundl((2*S+A+ noise1 )/2,1))
    else:
        St=df['St']
        
    if 'sav' in dolst.keys():
        sav=np.array([dolst['sav']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys():
        sav=np.array(helper.roundl((S+A+noise2)/3,0.5))
    else:
        sav=df['sav']
        
    if 'hous' in dolst.keys():
        hous=np.array([dolst['hous']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys():
        hous=np.array(helper.roundl((S+noise3)/2,0.5))
    else:
        hous=df['hous']
        
    if 'Cred' in dolst.keys():
        Cred=np.array([dolst['Cred']]*df.shape[0])
    elif 'A' in dolst.keys() or 'S' in dolst.keys():
        Cred=np.array(helper.roundl((0.5*S+1.5*A+ noise1 )/2,1))
    else:
        Cred=df['Cred']  

    #print (St)
    #X1 = (2*S+A+np.random.normal(loc=0.0, scale=0.2, size=N))#(np.random.normal(loc=100.0, scale=1.16, size=N)) # N(0,1)
    #X2 =(3*A+2*S+np.random.normal(loc=0.0, scale=0.2, size=N))#random_logit(100*X1)#+np.random.normal(loc=0.0, scale=0.16, size=N)
    #Y = random_logit((A+3*St+2*sav+hous)/50)
    Y = helper.roundlst((St/3+sav/3+hous/3+noise4))
    df=pd.DataFrame(np.array(list(S)),columns=['S'])
    

    df['A']=np.array(list(A))
    #print(df)
    df['S']=np.array(list(S))
    df['St']=np.array(list(St))
    df['sav']=np.array(list(sav))
    df['hous']=np.array(list(hous))
    df['Cred']=np.array(list(Cred))
    df['Y']=np.array(list(Y))
    
    return df
    #For each variable in dolst, change their descendants
        
        
