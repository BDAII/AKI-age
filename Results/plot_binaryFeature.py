# -*- coding: utf-8 -*-


import pickle
import pandas as pd
import numpy as np
import shap
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

#=======================================================
g=2 # age group
iD='days'
data_g = data[g-1]
wshap_g = wshap[g-1]  

shap.dependence_plot(iD, wshap_g.values, data_g.iloc[:,:-1], 
                     interaction_index=None)

#you can use xmax and xmin with a percentile notation to hide outliers
shap.dependence_plot(iD, wshap_g.values, data_g.iloc[:,:-1], 
                     interaction_index=None, xmin="percentile(1)", xmax="percentile(99)")

#shap.dependence_plot(iD, wshap_g.values, data_g.iloc[:,:-1], 
#                     interaction_index=None, xmin=0, xmax=7)


temp= data_g['label']
days = data_g['days']
aki_days = days[temp>0]
nonaki_days = days[temp==0]
        

#=======================================================
gd = []

for g in [1,2,3,4]:
    data_g = data[g-1]['gender']
    wshap_g = wshap[g-1]['gender']
    
    ind0 = data_g[data_g==0]
    ind1 = data_g[data_g==1]
    
    temp0 = list(ind0.index)
    temp1 = list(ind1.index)
    
    wshap0 = wshap_g[temp0]
    wshap1 = wshap_g[temp1]
    
    ws0 = np.array(wshap0)
    ws1 = np.array(wshap1)
    
    ws = [ws0,ws1]
    gd.append(ws)
    


def draw_gender(gd):
    fig, axes = plt.subplots(2,2,figsize=(10,8))

    #===g=1
    g=1
    axes[0,0].hist(gd[g-1][0],color='r',label='Female',alpha = 0.5)
    axes[0,0].hist(gd[g-1][1],color='b',label='Male',alpha = 0.5)
    axes[0,0].set_xlabel('Weight_shap for group 1')
    axes[0,0].legend(loc='best')


    #===g=2
    g=2
    axes[0,1].hist(gd[g-1][0],color='r',label='Female',alpha = 0.5)
    axes[0,1].hist(gd[g-1][1],color='b',label='Male',alpha = 0.5)
    axes[0,1].set_xlabel('Weight_shap for group 2')
    axes[0,1].legend(loc='best')


    #===g=3
    g=3
    axes[1,0].hist(gd[g-1][0],color='r',label='Female',alpha = 0.5)
    axes[1,0].hist(gd[g-1][1],color='b',label='Male',alpha = 0.5)
    axes[1,0].set_xlabel('Weight_shap for group 3')
    axes[1,0].legend(loc='best')

    #===g=4
    g=4
    axes[1,1].hist(gd[g-1][0],color='r',label='Female',alpha = 0.5)
    axes[1,1].hist(gd[g-1][1],color='b',label='Male',alpha = 0.5)
    axes[1,1].set_xlabel('Weight_shap for group 4')
    axes[1,1].legend(loc='best')

    plt.savefig(cd + '/feature_category/plot_shap_group/gender/hist/'+'Histogram for gender'
               +'.png',dpi=400,bbox_inches='tight')
    plt.savefig(cd + '/feature_category/plot_shap_group/gender/hist/'+'Histogram for gender'
               +'.pdf',dpi=400,bbox_inches='tight')    
  
    plt.show()


draw_gender(gd)

#======== ================================================================
#'race'

race1= []
for g in [1,2,3,4]:
    data_g = data[g-1]['race_1']
    wshap_g = wshap[g-1]['race_1']
    
    ind0 = data_g[data_g==0]
    ind1 = data_g[data_g==1]
    
    temp0 = list(ind0.index)
    temp1 = list(ind1.index)
    
    wshap0 = wshap_g[temp0]
    wshap1 = wshap_g[temp1]
    
    ws0 = np.array(wshap0)
    ws1 = np.array(wshap1)
    
    ws = [ws0,ws1]
    race1.append(ws)



race2= []
for g in [1,2,3,4]:
    data_g = data[g-1]['race_2']
    wshap_g = wshap[g-1]['race_2']
    
    ind0 = data_g[data_g==0]
    ind1 = data_g[data_g==1]
    
    temp0 = list(ind0.index)
    temp1 = list(ind1.index)
    
    wshap0 = wshap_g[temp0]
    wshap1 = wshap_g[temp1]
    
    ws0 = np.array(wshap0)
    ws1 = np.array(wshap1)
    
    ws = [ws0,ws1]
    race2.append(ws)



def draw_race(race1,race2):
    fig, axes = plt.subplots(4,2,figsize=(8,15))

    #===g=1 
    g=1
    axes[0,0].hist(race1[g-1][0],color='b',label='Other race',alpha = 0.5)
    axes[0,0].hist(race1[g-1][1],color='r',label='White',alpha = 0.5)
    axes[0,0].set_xlabel('Weight_shap for group 1')
    axes[0,0].legend(loc='best')

    axes[0,1].hist(race2[g-1][0],color='b',label='Other race',alpha = 0.5)
    axes[0,1].hist(race2[g-1][1],color='g',label='African American',alpha = 0.5)
    axes[0,1].set_xlabel('Weight_shap for group 1')
    axes[0,1].legend(loc='best')
    
    #===g=2
    g=2
    axes[1,0].hist(race1[g-1][0],color='b',label='Other race',alpha = 0.5)
    axes[1,0].hist(race1[g-1][1],color='r',label='White',alpha = 0.5)
    axes[1,0].set_xlabel('Weight_shap for group 2')
    axes[1,0].legend(loc='best')

    axes[1,1].hist(race2[g-1][0],color='b',label='Other race',alpha = 0.5)
    axes[1,1].hist(race2[g-1][1],color='g',label='African American',alpha = 0.5)
    axes[1,1].set_xlabel('Weight_shap for group 2')
    axes[1,1].legend(loc='best')


    #===g=3
    g=3
    axes[2,0].hist(race1[g-1][0],color='b',label='Other race',alpha = 0.5)
    axes[2,0].hist(race1[g-1][1],color='r',label='White',alpha = 0.5)
    axes[2,0].set_xlabel('Weight_shap for group 3')
    axes[2,0].legend(loc='best')

    axes[2,1].hist(race2[g-1][0],color='b',label='Other race',alpha = 0.5)
    axes[2,1].hist(race2[g-1][1],color='g',label='African American',alpha = 0.5)
    axes[2,1].set_xlabel('Weight_shap for group 3')
    axes[2,1].legend(loc='best')

    #===g=4
    g=4
    axes[3,0].hist(race1[g-1][0],color='b',label='Other race',alpha = 0.5)
    axes[3,0].hist(race1[g-1][1],color='r',label='White',alpha = 0.5)
    axes[3,0].set_xlabel('Weight_shap for group 4')
    axes[3,0].legend(loc='best')

    axes[3,1].hist(race2[g-1][0],color='b',label='Other race',alpha = 0.5)
    axes[3,1].hist(race2[g-1][1],color='g',label='African American',alpha = 0.5)
    axes[3,1].set_xlabel('Weight_shap for group 4')
    axes[3,1].legend(loc='best')

    plt.savefig(cd + '/feature_category/plot_shap_group/race/hist/'+'Histogram for race'
               +'.png',dpi=400,bbox_inches='tight')
    plt.savefig(cd + '/feature_category/plot_shap_group/race/hist/'+'Histogram for race'
               +'.pdf',dpi=400,bbox_inches='tight')    
  
    plt.show()


draw_race(race1,race2)


#===================================BMI ================================
'BMI'


data_g = data[g-1]
wshap_g = wshap[g-1]

ind = ['BMI_0','BMI_1','BMI_2','BMI_3','BMI_4']
fname = [x+str(dict_name[x]) for x in ind]

ws = wshap_g.loc[:,ind].values
dx = data_g.loc[:,ind].values

shap.summary_plot(ws,dx,feature_names=fname,max_display=len(ind),sort=False)


















































