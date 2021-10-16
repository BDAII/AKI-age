# -*- coding: utf-8 -*-


import pickle
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

#============ input data ================================
cd = '.../AKI_age/shap_importance/'

allmed = pd.read_csv(cd+'/med_mining/allmed_dropDuplicates.csv')
del allmed['Unnamed: 0']

med_dir = pd.read_csv(cd+'/med_mining/get_all_med.csv')
med_dir = med_dir.loc[:,['medID','group']]


f = open(cd+'final_data_wshap.pkl','rb')
Data = pickle.load(f)
f.close()

data = Data['data']
wshap = Data['wshap']

#================function ===============
'''
Check whether the shap_value distribution is the same between taking the medicine and not taking the medicine, 
and calculate the P value;
Kolmogorov-smirnov test;
To judge risk or protection by calculating the mean() of shap_value obtained by taking medicine versus not taking medicine? 
'''

def get_direction_pvalue(g,ID):
    data_g = data[g-1]
    wshap_g = wshap[g-1]
    
    temp = data_g.loc[:,ID] 
    # temp = data_g.iloc[:,-1] 
    temp0 = temp[temp==0]
    temp0 = list(temp0.index)
    temp1 = temp[temp>0]
    temp1 = list(temp1.index)

    
    temp3 = wshap_g.loc[temp1,ID]
    temp4 = wshap_g.loc[temp0,ID] 
    
    pos_effect = np.mean(temp3)
    neg_effect = np.mean(temp4)  

    temp5 = np.array(temp3)
    temp6 = np.array(temp4)
    
    [s,pvalue]=ks_2samp(temp5,temp6)
    
    return pos_effect, neg_effect, pvalue



def compute_group(g,ID):
    temp0 = med_dir[med_dir['medID']==ID]
    temp1 = temp0['group']
    if g in temp1.values:
        pos_effect, neg_effect, pvalue = get_direction_pvalue(g,ID)
    else:
        pvalue = np.nan
        pos_effect = np.nan
        neg_effect = np.nan
    return pos_effect, neg_effect, pvalue


#============= main compute ==============
allMED = pd.DataFrame(columns=['medID','medName','g1_1','g1_0','g1_p','g2_1','g2_0','g2_p',
                               'g3_1','g3_0','g3_p','g4_1','g4_0','g4_p'])


for i in range(len(allmed)):
    i_medID = allmed['medID'][i]
    i_medName = allmed['medName'][i]
    
    [g11,g12,g13] = compute_group(1,i_medID)
    [g21,g22,g23] = compute_group(2,i_medID)
    [g31,g32,g33] = compute_group(3,i_medID)
    [g41,g42,g43] = compute_group(4,i_medID)
    
    allMED = allMED.append(pd.DataFrame({'medID':[i_medID],'medName':[i_medName],'g1_1':[g11],'g1_0':[g12],'g1_p':[g13],
                                         'g2_1':[g21],'g2_0':[g22],'g2_p':[g23],
                               'g3_1':[g31],'g3_0':[g32],'g3_p':[g33],'g4_1':[g41],'g4_0':[g42],'g4_p':[g43]}),
                           ignore_index=True)


#====== save ===========
allMED.to_excel(cd+'/med_mining/med166_shap_direction_pvalue.xlsx', index=False,float_format='%.4f')









