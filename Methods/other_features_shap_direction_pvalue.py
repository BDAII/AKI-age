# -*- coding: utf-8 -*-


import pickle
import pandas as pd
#from scipy.stats import fisher_exact
import numpy as np
from scipy.stats import ks_2samp

cd = '.../AKI_age/shap_importance/'

f = open(cd+'final_data_wshap.pkl','rb')
Data= pickle.load(f)
f.close()

data = Data['data']
wshap = Data['wshap']

############### map name ###############
csv_map = pd.read_csv(cd+'AKI_age_map.csv')

del csv_map['Index']
dict_name = csv_map.set_index('ID').T.to_dict('list')


###########
g=4

data0 = data[g-1]
wshap0 = wshap[g-1]

#================ rank ==================
k=150

temp = np.abs(wshap0).mean(0)
ind = temp.sort_values(ascending=False)

temp1 = ind[:k].index
df_wshap = wshap0.loc[:,temp1]


#=================== get med index==============
def get_other_ind(X0):
    col_names = list(X0.columns)
    other_ind = []
    for i in range(len(col_names)):
        col_name = col_names[i]
        if 'MED' in col_name[:3]:
            continue
        elif 'day' in col_name[:3]:
            continue
        else:            
            other_ind.append(col_name)
    return  other_ind


other_ind = get_other_ind(df_wshap)


otherID = [x for x in other_ind]
otherName = [str(dict_name[x]) for x in other_ind]


def get_direction_pvalue(ID):    
    temp = data0.loc[:,ID]
    temp0 = temp[temp==0]
    temp0 = list(temp0.index)
    temp1 = temp[temp>0]
    temp1 = list(temp1.index)

    
    temp3 = wshap0.loc[temp1,ID]
    temp4 = wshap0.loc[temp0,ID] 
    
    pos_effect = np.mean(temp3)
    neg_effect = np.mean(temp4)  

    temp5 = np.array(temp3)
    temp6 = np.array(temp4)
    
    [s,pvalue]=ks_2samp(temp5,temp6)
    
    return pos_effect, neg_effect, pvalue


def compute_shap_direction(other_ID):
    results = pd.DataFrame(columns=['pos_shap','neg_shap','pvalue'])
    for i in range(len(other_ID)):
        rt = get_direction_pvalue(other_ID[i])
        results = results.append(pd.DataFrame({'pos_shap':[rt[0]],'neg_shap':[rt[1]],'pvalue':[rt[2]]}),
                                    ignore_index=True)

    return results



otherPP = compute_shap_direction(otherID)
df_otherID = pd.DataFrame(otherID,columns=['otherID'])
df_otherName = pd.DataFrame(otherName,columns=['Name'])
df = pd.concat([df_otherID,df_otherName,otherPP],axis=1)
df.to_excel(cd+'/feature_category/other_features_shap_direction_pvalue/otherFeatures_direction_pvalue_group'+str(g)+'.xlsx')











