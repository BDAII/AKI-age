# -*- coding: utf-8 -*-


import pandas as pd


cd = '.../AKI_age/shap_importance/'

df = pd.DataFrame(columns=['medID', 'medName', 'group', 'OR', 'LCI', 'UCI', 'p_value', 'h'])

for g in [0,1,2,3]:
    temp0 = pd.read_csv(cd+'/med_OR/med_OR_group'+str(g+1)+'.csv')
    del temp0['Unnamed: 0']
    temp1 = temp0.iloc[:,:2]
    temp2 = pd.DataFrame([g+1]*temp1.shape[0], columns = ['group'])
    temp3 = temp0.iloc[:,6:]
    temp4 = pd.concat([temp1,temp2,temp3],axis=1)
    df = pd.concat([df,temp4],axis=0)
    

df.to_csv(cd+'med_mining/get_all_med.csv')

#==================================
dff = df.iloc[:,:2]
    
df_dropDuplicates = dff.drop_duplicates()    
    
df_dropDuplicates.to_csv(cd + 'med_mining/allmed_dropDuplicates.csv')    
    
    
#===================================
dff0 = df.iloc[:,[0,1,-1]]

dff0 = dff0[dff0['h'].isin([1])]    
df_dropORp_dropDuplicates = dff0.drop_duplicates()  
  
del  df_dropORp_dropDuplicates['h']   

df_dropORp_dropDuplicates.to_csv(cd + 'med_mining/allmed_dropORp_dropDuplicates.csv')      
    
    