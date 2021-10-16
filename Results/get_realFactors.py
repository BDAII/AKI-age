# -*- coding: utf-8 -*-


import pandas as pd
#from scipy.stats import fisher_exact
import numpy as np
import pickle 
from scipy import stats

cd = '.../AKI_age/shap_importance/'

f = open(cd+'final_data_wshap.pkl','rb')
Data= pickle.load(f)
f.close()

############### map name ###############
csv_map = pd.read_csv(cd+'AKI_age_map.csv')

############################
def get_realRiskFactors(Data, g, k):
    data = Data['data']
    wshap = Data['wshap']

    del csv_map['Index']
    dict_name = csv_map.set_index('ID').T.to_dict('list')

    data0 = data[g]
    wshap0 = wshap[g]
    #================ rank ==================
    temp = np.abs(wshap0).mean(0)
    ind = temp.sort_values(ascending=False)
   
    temp1 = ind[:k].index
    df_wshap = wshap0.loc[:,temp1]

    def chi2(x,y):
        crosstab = pd.crosstab(x,y)
        temp = stats.chi2_contingency(crosstab)
        pvalue = np.round(temp[1],4)
        if pvalue<0.05:
            h = 1
        else:
            h = 0
            
        return pvalue,h

    def get_direction(ID):    
        temp = data0.loc[:,ID]
        temp1 = temp[temp>0]
        temp1 = list(temp1.index)    
        temp3 = df_wshap.loc[temp1,ID] # or wshap0   
        pos_effect = np.mean(temp3)
    
        return pos_effect


    yy = data0.loc[:,'label']

    finID  = []
    finwshap = []
    for i in list(df_wshap.columns):
        # chi2
        xx = data0.loc[:,i]
        pvalue, h = chi2(xx,yy)
        if h == 1:
            pos_effect = get_direction(i)
            if pos_effect > 0.1:
                finID.append(i)
                finwshap.append(pos_effect)

    # save
    df = pd.DataFrame({'finID':finID,'finwshap':finwshap})            
    df.to_excel('.../AKI_age/realFactors/G'+str(g)+'.xlsx')
    
    return df



for g in range(4):
    get_realRiskFactors(Data, g, 150)

