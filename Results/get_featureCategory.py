# -*- coding: utf-8 -*-


import pandas as pd
#from scipy.stats import fisher_exact
import numpy as np
import pickle 
from scipy import stats
import shap

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
def find_FC(Data,g,k):
    data0 = data[g-1]
    wshap0 = wshap[g-1]
    #================ rank ==================
    temp = np.abs(wshap0).mean(0)
    ind = temp.sort_values(ascending=False)

    temp1 = ind[:k].index
    df_wshap = wshap0.loc[:,temp1]
    df_data = data0.loc[:,temp1]

    ############ filter ################
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
        effect = np.round(np.mean(temp3),3)
    
        return effect


    yy = data0.loc[:,'label']

    finID  = []
    finwshap = []
    
    for i in list(df_wshap.columns):
        xx = data0.loc[:,i]
        pvalue, h = chi2(xx,yy)
        if h == 1:
            effect = get_direction(i)
            if (effect > 0.05) | (effect < -0.05):
                finID.append(i)
                finwshap.append(effect)


    finID = sorted(finID,key=str.lower)

    #=================== get index==============
    def get_ind(finID):
        col_names = finID
        demo_vital_ind = []
        lab_ind = []
        drg_ind = []
        ccs_ind = []
        med_ind = []
        for i in range(len(col_names)):
            col_name = col_names[i]
            if 'Lab' in col_name[:3]:
                lab_ind.append(col_name)
            elif 'DRG' in col_name[:3]:
                drg_ind.append(col_name)
            elif 'CCS' in col_name[:3]:
                ccs_ind.append(col_name)
            elif 'MED' in col_name[:3]:
                med_ind.append(col_name)
            else:
                if 'days' not in col_name[:4]:
                    demo_vital_ind.append(col_name)
                
        return demo_vital_ind, lab_ind, drg_ind, ccs_ind, med_ind

    demo_vital_ind, lab_ind, drg_ind, ccs_ind, med_ind = get_ind(finID)
    
    return finID, demo_vital_ind, lab_ind, drg_ind, ccs_ind, med_ind

################################################################
g = 2; k = 150; # for example
finID, demo_vital_ind, lab_ind, drg_ind, ccs_ind, med_ind = find_FC(Data,g,k)
#===========
ind = demo_vital_ind
fname = [x+str(dict_name[x]) for x in ind]
s = df_wshap.loc[:,ind].values
f = df_data.loc[:,ind].values
shap.summary_plot(s, f,feature_names=fname,max_display=len(ind),color_bar=False)

#==============
ind = lab_ind
fname = [x+str(dict_name[x]) for x in ind]
s = df_wshap.loc[:,ind].values
f = df_data.loc[:,ind].values
shap.summary_plot(s, f,feature_names=fname,max_display=len(ind),color_bar=False)

#==============
ind = drg_ind
fname = [x+str(dict_name[x]) for x in ind]
s = df_wshap.loc[:,ind].values
f = df_data.loc[:,ind].values
shap.summary_plot(s, f,feature_names=fname,max_display=len(ind),color_bar=False)

#==============
ind = ccs_ind
fname = [x+str(dict_name[x]) for x in ind]
s = df_wshap.loc[:,ind].values
f = df_data.loc[:,ind].values
shap.summary_plot(s, f,feature_names=fname,max_display=len(ind),color_bar=False)

#==============
ind = med_ind
fname = [x+str(dict_name[x]) for x in ind]
s = df_wshap.loc[:,ind].values
f = df_data.loc[:,ind].values
#shap.summary_plot(s, f,feature_names=fname,max_display=len(ind),color_bar=False)

f1=f
f1[f1>0]=1
shap.summary_plot(s, f1,feature_names=fname,max_display=len(ind),color_bar=False)


# save
df = pd.DataFrame({'finID':finID,'finwshap':finwshap})            
df.to_excel('.../AKI_age/realFactors/G'+str(g)+'.xlsx')
























