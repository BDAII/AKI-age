# -*- coding: utf-8 -*-


import pandas as pd
import pickle
import numpy as np


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


#================================

def personalized(g,s,k):
    data_g = data[g-1]
    wshap_g = wshap[g-1]
    
    s_data = data_g.iloc[s-1,:]
    s_wshap = wshap_g.iloc[s-1,:]

    pos_wshap = s_wshap[s_wshap>0]
    neg_wshap = s_wshap[s_wshap<0]

    pos_wshap = pos_wshap.sort_values(ascending = False)
    neg_wshap = neg_wshap.sort_values()
    
    temp_pos = list(pos_wshap.index)[:k]
    temp_neg = list(neg_wshap.index)[:k]

    name_pos = [x+str(dict_name[x]) for x in temp_pos]
    name_neg = [x+str(dict_name[x]) for x in temp_neg]

    value_pos = [s_data[x] for x in temp_pos]
    value_neg = [s_data[x] for x in temp_neg]    
        
    rank = np.arange(1,k+1,1).tolist()

    df = pd.DataFrame({'Rank':rank,'Name_pos':name_pos,'Values_pos':value_pos,'weight_shap_pos':pos_wshap.values[:k],
                       'Name_neg':name_neg,'Values_neg':value_neg,'weight_shap_neg':neg_wshap.values[:k]})

    df.to_csv(cd+'/personalized_interpretation/G'+str(g)+'_s'+str(s)+'.csv',float_format='%.4f')    


#=================    
# for example 
g=1;s=1;k=10
personalized(g,s,k)    
   
g=1;s=0;k=10
personalized(g,s,k) 

   
g=4;s=1;k=10
personalized(g,s,k)    
 
g=4;s=0;k=10
personalized(g,s,k)











