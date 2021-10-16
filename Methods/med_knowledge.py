# -*- coding: utf-8 -*-


import pickle
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import shap
#import networkx as nx
#import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import os
import matplotlib.pyplot as plt


cd = '.../AKI_age/shap_importance/'

#============ input data ===============
f = open(cd+'final_data_wshap.pkl','rb')
Data = pickle.load(f)
f.close()

data = Data['data']
wshap = Data['wshap']

############### map name ###############
csv_map = pd.read_csv(cd+'AKI_age_map.csv')

del csv_map['Index']
dict_name = csv_map.set_index('ID').T.to_dict('list')


#=============functionn =====================
def draw_shap(g,ID):
    data_g = data[g-1]
    wshap_g = wshap[g-1]    
    shap.dependence_plot(ID, wshap_g.values, data_g.iloc[:,:-1], interaction_index=None)
    
#Kolmogorov-Smirnov test

def get_pvalue(g,ID):
    data_g = data[g-1]
    wshap_g = wshap[g-1]
    
    temp = data_g.iloc[:,-1]
    temp0 = temp[temp==0]
    temp0 = list(temp0.index)
    temp1 = temp[temp>0]
    temp1 = list(temp1.index)

    
    temp3 = wshap_g.loc[temp1,ID]
    temp4 = wshap_g.loc[temp0,ID] 
    
    temp5 = np.array(temp3)
    temp6 = np.array(temp4)
    
    [s,pvalue]=ks_2samp(temp5,temp6)
    
    return pvalue


   

def get_direction(g,ID):
    data_g = data[g-1]
    wshap_g = wshap[g-1]
    
    temp = data_g.loc[:,ID]
    temp0 = temp[temp==0]
    temp0 = list(temp0.index)
    temp1 = temp[temp>0]
    temp1 = list(temp1.index)

    
    temp3 = wshap_g.loc[temp1,ID]
    temp4 = wshap_g.loc[temp0,ID] 
    
    pos_effect = np.mean(temp3)
    neg_effect = np.mean(temp4)  
    
    return pos_effect, neg_effect



def get_cosSim(x,y):
    myx=np.array(x)
    myy=np.array(y)
    cos1=np.sum(myx*myy)
    cos21=np.sqrt(sum(myy*myy))
    cos22=np.sqrt(sum(myx*myx))
    temp = cos1/float(cos22*cos21)
    
    return temp


def get_correlation(g,ID):
    data_g = data[g-1]
    
    result_cosSim = pd.Series(index=data_g.columns)
    for i in data_g.columns:
        temp = get_cosSim(data_g.loc[:,i], data_g.loc[:,ID])
        result_cosSim[i] = temp
        
    finalD = result_cosSim.sort_values(ascending=False)[1:]
    
    return finalD


def get_OR_P(df_a, df_b):
    temp0 = df_a[:]
    temp0[temp0[:]>0]=1
    tab = pd.crosstab(temp0, df_b)
    a = tab.iloc[0,0]
    b = tab.iloc[0,1]
    c = tab.iloc[1,0]
    d = tab.iloc[1,1]
    rt_OR, rt_p = fisher_exact(tab)
    if rt_p <=0.05:
        rt_h = 1
    else:
        rt_h = 0
    
    temp1 = 1.96*np.sqrt(1./a + 1./b + 1./c + 1./d)
    temp2 = np.log(rt_OR)
    rt_lci = np.exp(temp2 - temp1)
    rt_uci = np.exp(temp2 + temp1)
    
    rst_OR = [rt_OR,rt_lci,rt_uci,rt_p,rt_h,a,b,c,d]
    
    return rst_OR


def compute_OR(g,ID):
    data0 = data[g-1]
    finD = get_OR_P(data0.loc[:,ID],data0.loc[:,'label']) 
    
    return finD



def get_pect(g,c,d):
    if g==1:
        rt = (c+d)/12873 # sample number in age group 1
    elif g==2:
        rt = (c+d)/25197
    elif g==3:
        rt = (c+d)/18098
    elif g==4:
        rt = (c+d)/20789
    else:
        print('wrong g!')
    
    return rt


    
#================================================= 
#====================
g=1
iD='gender'

file_path = cd + '/med_knowledge/plot_shap/' +iD

if not os.path.exists(file_path):
    os.mkdir(file_path)
    
#========== plot shap value ===========
draw_shap(g,iD)  

#==========correlation analysis =========
cosSim = get_correlation(g,iD)
temp = list(cosSim.index)
name = [str(dict_name[x]) for x in temp]
cosSimA = pd.DataFrame({'ID':cosSim.index,'name':name,'cosSim':cosSim.values})
cosSimA.to_csv(cd+'/med_knowledge/plot_shap/'+iD+'/G'+str(g)+'_'+iD+'_cosSim.csv',float_format='%.4f')

#============compute p value=====
pvalue = get_pvalue(g,iD)    

#============ compute direction =========
[pos_effect, neg_effect] = get_direction(g,iD)

#compute_OR(g,iD)
rt_OR = compute_OR(g,iD)

pect = get_pect(g,rt_OR[7],rt_OR[8])

finD = [pos_effect, neg_effect,pvalue,rt_OR,pect]


def mainfunction(g,iD):
    file_path = cd + '/med_knowledge/plot_shap/' +iD
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    draw_shap(g,iD)  
    cosSim = get_correlation(g,iD)
    temp = list(cosSim.index)
    name = [str(dict_name[x]) for x in temp]
    cosSimA = pd.DataFrame({'ID':cosSim.index,'name':name,'cosSim':cosSim.values})
    cosSimA.to_csv(cd+'/med_knowledge/plot_shap/'+iD+'/G'+str(g)+'_'+iD+'_cosSim.csv',float_format='%.4f')


mainfunction(1,'MED975') # for example















