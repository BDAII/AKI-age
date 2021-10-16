# -*- coding: utf-8 -*-


import pickle
import pandas as pd

import numpy as np
#import networkx as nx
#import matplotlib.pyplot as plt

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
    

#========== plot shap value ===========

def compute_dot(g,iD):
    d = data[g-1].loc[:,iD]
    ws = wshap[g-1].loc[:,iD]
    
    days = [i for i in range(8)]
    
    ws_all = []
    ws_mean = []
    ws_std =[]
    for i in days:
      temp1 = ws[(d==i)] 
      temp2 = np.mean(temp1)
      temp3 = np.std(temp1)
      ws_all.append(temp1)
      ws_mean.append(temp2)
      ws_std.append(temp3)
      
    return days, ws_all, ws_mean, ws_std


#days, ws_all, ws_mean, ws_std = compute_dot(g,iD)

def plot_dots(g,iD):
    days, ws_all, ws_mean, ws_std = compute_dot(g,iD)
    fig = plt.figure(figsize=(5,3))
    
    for i in days:
        xi = np.full(np.size(ws_all[i]),i) 
        if i==0:
            plt.scatter(xi,ws_all[i],alpha=0.1)
        else:
            plt.scatter(xi,ws_all[i],alpha=0.1)
        
    plt.errorbar(days,ws_mean,yerr=ws_std,fmt='s',ecolor='r',color='k',elinewidth=3,capsize=4)
    plt.plot(days,ws_mean,'olive')
    plt.xlabel('Days of medication in the first 7 days')
    plt.ylabel('Weighed SHAP value')
    plt.title('Group '+str(g)+': '+iD+str(dict_name[iD]))
    plt.savefig(file_path +'/figure_'+iD+'.png',dpi=400,bbox_inches='tight')
    plt.show()


def plot_dot(days, ws_all, ws_mean, ws_std):
    #fig = plt.figure(figsize=(5,3))
    
    for i in days:
        xi = np.full(np.size(ws_all[i]),i) 
        if i==0:
            plt.scatter(xi,ws_all[i],alpha=0.1)
        else:
            plt.scatter(xi,ws_all[i],alpha=0.1)
        
    plt.errorbar(days,ws_mean,yerr=ws_std,fmt='s',ecolor='r',color='k',elinewidth=3,capsize=4)
    plt.plot(days,ws_mean,'olive')
    plt.xlabel('Days of medication in the first 7 days')
    plt.ylabel('Weighed SHAP value')
    plt.title('Group '+str(g)+': '+iD+str(dict_name[iD]))
    #plt.savefig()
    
def draw_figure(iD):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(221)
    g = 1
    days, ws_all, ws_mean, ws_std = compute_dot(g,iD)
    plot_dot(days, ws_all, ws_mean, ws_std)
    
    ax = fig.add_subplot(222)
    g = 2
    days, ws_all, ws_mean, ws_std = compute_dot(g,iD)
    plot_dot(days, ws_all, ws_mean, ws_std)

    ax = fig.add_subplot(223)
    g = 3
    days, ws_all, ws_mean, ws_std = compute_dot(g,iD)
    plot_dot(days, ws_all, ws_mean, ws_std)

    ax = fig.add_subplot(224)
    g = 4
    days, ws_all, ws_mean, ws_std = compute_dot(g,iD)
    plot_dot(days, ws_all, ws_mean, ws_std)
    
    plt.subplots_adjust(wspace=0.3,hspace=0.5)
    plt.savefig(file_path +'/figure_'+iD+'.png',dpi=400,bbox_inches='tight')
    plt.show()

#====================

#for four groups
iD='MED464'

file_path = cd + '/med_knowledge/plot_med07/' +iD

if not os.path.exists(file_path):
    os.mkdir(file_path)


draw_figure(iD)

# for only one group
g = 2
iD ='MED510'#'MED478''MED1146''MED572''MED510'

file_path = cd + '/med_knowledge/plot_med07/' +iD

if not os.path.exists(file_path):
    os.mkdir(file_path)

plot_dots(g,iD)














