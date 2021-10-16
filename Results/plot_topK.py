# -*- coding: utf-8 -*-


import pickle
import pandas as pd
import shap

cd = '.../AKI_age/shap_importance/'


f = open(cd+'final_weightSHAP.pkl','rb')
wshap = pickle.load(f)
f.close()


f = open(cd+'Data.pkl','rb')
data = pickle.load(f)
f.close()


############### map name ###############
csv_map = pd.read_csv(cd+'AKI_age_map.csv')

del csv_map['Index']
dict_name = csv_map.set_index('ID').T.to_dict('list')



###########################################################################
 
def topK_summary_plot(d,k):
    
    temp = data[d]
    x = temp.iloc[:,:-1]

    temp0 = wshap[d].abs()
    sum_shap = temp0.mean(axis=0)
    sort_shap = sum_shap.sort_values(ascending=False)
    sort_shap_ind = list(sort_shap.index)
       
    ind = sort_shap_ind[:k]
    
    fname = [x+str(dict_name[x]) for x in ind]
    
    s = wshap[d].loc[:,ind].values
    f = x.loc[:,ind].values
    shap.summary_plot(s,f,feature_names=fname,max_display=len(ind)) #,plot_type='bar'
    

d = topK_summary(g,k)


