import time
import copy
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from mip import Model, xsum, maximize, minimize, BINARY

def get_query_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c,g_Ac_lst,interference, blocks, backdoor):
    
    #Backdoor list
    # backdoorlst = list(set(attr for a in Ac for attr in backdoor[a]))
    backdoorlst = []
    for attr in Ac:
        backdoorlst.extend(backdoor[attr])
    backdoorlst=list(set(backdoorlst))

    #Combination Backdoor 
    if backdoorlst:
        backdoorvals = get_C_set(df, backdoorlst)
    else:
        backdoorvals = [] if q_type == 'count' else [[]] #else -> avg
    
    total_prob=0
    regr=''
    iter=0

    for backdoorvallst in backdoorvals:
        conditioning_set = prelst + Ac + backdoorlst
        conditioning_val = prevallst + c + backdoorvallst

        if iter == 0:
            if q_type == 'count':
                start = time.process_time() 
                regr = train_regression(df, conditioning_set, conditioning_val, postlst, postvallst)
            elif q_type == 'avg':
                regr = train_regression_raw(df, conditioning_set, conditioning_val, AT)

        pogivenck = regr.predict([conditioning_val])[0]
        pcgivenk = get_prob_o_regression(df, prelst, prevallst, backdoorlst, backdoorvallst)
        total_prob += pogivenck * pcgivenk
        iter += 1
    return total_prob



def get_C_set(df,C):
    lst=[]
    for Cvar in C:
        lst.append(list(set(list(df[Cvar]))))
        
    combination_lst= (get_combination(lst,[]))
    
    return combination_lst

def get_combination(lst,tuplelst):
    i=0
    new_tuplelst=[]
    if len(tuplelst)==0:
        l=lst[0]
        for v in l:
            new_tuplelst.append([v])
        if len(lst)>1:
            return get_combination(lst[1:],new_tuplelst)
        else:
            return new_tuplelst
    

    currlst=lst[0]
    for l in tuplelst:
        
        for v in currlst:
            newl=copy.deepcopy(l)
            newl.append(v)
            new_tuplelst.append(newl)
        
    if len(lst)>1:
        return get_combination(lst[1:],new_tuplelst)
    else:
        return new_tuplelst


def train_regression_raw(df,conditional,conditional_values,AT):
    new_lst=[]
    count=0

    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    #regr = RandomForestRegressor(random_state=0)
    regr = LinearRegression()#random_state=0)
    regr.fit(X, df[AT])
    return regr

def train_regression(df,conditional,conditional_values,target,target_val):
    new_lst=[]
    count=0
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
        if new_lst[-1]==1:
            count+=1
    if len(conditional)==0:
        return count*1.0/df.shape[0]
    if len(list(set(new_lst)))==1:
        if new_lst[0]==1:
            return 1
        else:
            return 0
        
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    regr = RandomForestRegressor(random_state=0)
    #regr = LogisticRegression(random_state=0)
    regr.fit(X.values, new_lst)
    return regr

def get_prob_o_regression(df,conditional,conditional_values,target,target_val):
    new_lst=[]
    count=0
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
        if new_lst[-1]==1:
            count+=1
    if len(conditional)==0:
        return count*1.0/df.shape[0]
    if len(list(set(new_lst)))==1:
        if new_lst[0]==1:
            return 1
        else:
            return 0
        
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    start = time.process_time()

    regr = RandomForestRegressor(random_state=0)
    #regr = LogisticRegression(random_state=0)
    regr.fit(X, new_lst)
    return regr.predict([conditional_values])[0]
    #return(regr.predict_proba([conditional_values])[0][1])
  
def get_val(row,target,target_val):
    i=0
    while i<len(target):
        if not int(row[target[i]])==int(target_val[i]):
            return 0
        i+=1
    return 1


def get_logistic_param(df,conditional,target,target_val):
    new_lst=[]
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
    X=df[conditional]
    regr = LinearRegression()
    regr.fit(X, df[target])
    print (regr.coef_)
    print (regr.intercept_)
    return (regr.coef_.tolist()[0],[regr.intercept_])


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

def get_data(N,seed,num_bins):
    #N=10000
    np.random.seed(seed)
    S=np.random.binomial(n=1, p=0.5,size=N)
    A=np.random.binomial(n=2, p=0.5,size=N)
    noise1=np.random.binomial(n=2, p=0.01,size=N)
    noise2=np.random.binomial(n=2, p=0.01,size=N)
    noise3=np.random.binomial(n=2, p=0.01,size=N)
    noise4=np.random.binomial(n=2, p=0.01,size=N)
    St=((2*S+A+ noise1 )/2)
    Cred=np.random.binomial(n=2, p=0.21,size=N)#np.array(((0.5*S+1.5*A+ noise1 )/2))
    
    sav=np.array(((S+A+noise2)/3))
            
    hous=np.array(((S+noise3)/2))
    bins={}
    #X1 = (2*S+A+np.random.normal(loc=0.0, scale=0.2, size=N))#(np.random.normal(loc=100.0, scale=1.16, size=N)) # N(0,1)
    #X2 =(3*A+2*S+np.random.normal(loc=0.0, scale=0.2, size=N))#random_logit(100*X1)#+np.random.normal(loc=0.0, scale=0.16, size=N)
    #Y = random_logit((A+3*St+2*sav+hous)/50)
    Y = ((A+St/3+sav/3+hous/3+noise4))#+Cred*0.01))
    df=pd.DataFrame(S,columns=['S'])
    df_U=pd.DataFrame(S,columns=['S'])
    df_U['noise1']=noise1
    df_U['noise2']=noise2
    df_U['noise3']=noise3
    df_U['noise4']=noise4
    df['A']=A
    df['S']=S
    df['St_orig']=St
    df['Cred_orig']=Cred
    df['sav_orig']=sav
    df['hous_orig']=hous
    (df['St'],bins['St']) = bucketize(St,num_bins)
    (df['Cred'],bins['Cred']) = bucketize(Cred,num_bins)
    (df['sav'],bins['sav']) = bucketize(sav,num_bins)
    (df['hous'],bins['hous']) = bucketize(hous,num_bins)
    df['Y']=Y
    
    
    return (df,df_U,bins)

def optimization(df,A,aval,Adomain,klst,kval,alpha,betalst,beta0, bins):
    
    backdoorvals=get_C_set(df,klst)
    print (backdoorvals,len(A),len(betalst))
    
    betalst_backdoor=betalst[len(A):]
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
        print("shapes",sampled_df.shape[0],df.shape[0],tmpsum)
    print (sum_backdoor)
     
    
    
    m = Model("Test")
    i=0
    var_lst=[]
    var_map={}
    while i<len(A):
        j=0
        while j<len(Adomain[i]):

            var_lst.append(m.add_var(var_type=BINARY))
            var_map[len(var_lst)-1]=(i,j)
            j+=1
        i+=1
    print ("beta list is ",betalst,beta0)
    cost_lst=[]
    constr_lst=[]
    constr_lst.append(beta0)
    iter=0
    i=0
    while i<len(A):
        j=0
        del_cons=[]
        while j<len(Adomain[i]):
            '''
            if Adomain[i][j]==aval[i]:
                constr_lst.append(betalst[i]*Adomain[i][j])
                j+=1
                continue
            '''
            constr_lst.append(betalst[i]*var_lst[iter]*(Adomain[i][j]))
            del_cons.append(var_lst[iter])
            iter+=1
            j+=1
        m += xsum(del_cons) <= 1
        i+=1
    constr_lst.append(sum_backdoor)
    '''i=len(A)
    while i<len(betalst):
        constr_lst.append(betalst[i]*kval[i-len(A)])
        i+=1
    '''

    print (constr_lst)
    #m+=xsum(constr_lst)>=math.log(alphak*1.0/(1-alphak))
    m.objective = maximize(xsum(constr_lst))
    m.optimize()
    if m.num_solutions:
        print('Objective value %g found:'
                  % (m.objective_value))
        i=0
        score=0
        gt_score=0
        while i<len(var_lst):
            print (i,var_lst[i].x,var_map[i], Adomain[var_map[i][0]][var_map[i][1]])
            if var_lst[i].x==1:
                vallst=[]
                if var_map[i][0]==0:
                    vallst=bins['St']
                elif var_map[i][0]==1:
                    vallst=bins['sav']
                elif var_map[i][0]==2:
                    vallst=bins['hous']
                else:
                    vallst=bins['Cred']
                print (vallst,(vallst[var_map[i][1]],vallst[var_map[i][1]+1]))
                print ("sc",betalst[var_map[i][0]]*((vallst[var_map[i][1]]+vallst[var_map[i][1]+1])/2))
                score+= betalst[var_map[i][0]]*((vallst[var_map[i][1]]+vallst[var_map[i][1]+1])/2)
                gt_score+=betalst[var_map[i][0]]*((vallst[-1]))
            i+=1
        print ("sum backdoor",sum_backdoor,score) 
        print ("score is ",score+sum_backdoor+beta0,gt_score+sum_backdoor+beta0)
        print ("score is ",score,sum_backdoor,beta0)#,gt_score+sum_backdoor+beta0)
        return (score+sum_backdoor+beta0)*1.0/(gt_score+sum_backdoor+beta0),(gt_score+sum_backdoor+beta0)
    