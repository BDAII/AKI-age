# -*- coding: utf-8 -*-


import pickle
import pandas as pd
from scipy.stats import fisher_exact
import numpy as np

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
g=3

data0 = data[g]
wshap0 = wshap[g]
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

#============================================

def get_OR_P(df_a, df_b):
    temp0 = df_a[:]
#    temp0[temp0[:]>0]=1
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
    
    finData = [a,b,c,d,rt_OR,rt_lci,rt_uci,rt_p,rt_h]
    return finData
    



def compute_OR_p(other_ID):
    results = pd.DataFrame(columns=['a','b','c','d','OR','LCI','UCI','p_value','h'])
    for i in range(len(other_ID)):
        rt = get_OR_P(data0.loc[:,other_ID[i]],data0.loc[:,'label'])
        results = results.append(pd.DataFrame({'a':[rt[0]],'b':[rt[1]],'c':[rt[2]],'d':[rt[3]],
                                                  'OR':[rt[4]],'LCI':[rt[5]],'UCI':[rt[6]],'p_value':[rt[7]],'h':[rt[8]]}),
                                    ignore_index=True)

    return results



otherOR = compute_OR_p(otherID)
df_otherID = pd.DataFrame(otherID,columns=['otherID'])
df_otherName = pd.DataFrame(otherName,columns=['Name'])
df = pd.concat([df_otherID,df_otherName,otherOR],axis=1)
df.to_csv(cd+'/feature_category/other_features_OR/otherFeatures_OR_group'+str(g+1)+'.csv')
