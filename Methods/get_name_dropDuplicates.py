# -*- coding: utf-8 -*-


import pandas as pd

cd = '.../AKI_age/shap_importance/get_top150_name/'

df = pd.DataFrame(columns=['ID', 'Name'])

for g in [1,2,3,4]:
    temp0 = pd.read_excel(cd+'top150_group'+str(g)+'.xlsx')
    temp1 = temp0.loc[:,['ID','Name']]
    df = pd.concat([df,temp1],axis=0)

#######
df_dropDuplicates = df.drop_duplicates()    

df_dropDuplicates = df_dropDuplicates.reset_index(drop=True)
   
df_dropDuplicates.to_excel(cd + '/dropDuplicates/allName_dropDuplicates.xlsx') 


#######################
for i in df_dropDuplicates.index:
    if df_dropDuplicates.loc[i , 'ID'].startswith("MED",0,3):
        df_dropDuplicates.drop(index=i,axis=0,inplace=True)   

df_dropDuplicates = df_dropDuplicates.reset_index(drop=True)    

df_dropDuplicates.to_excel(cd + '/dropDuplicates/binaryF_dropDuplicates.xlsx') 


##########################
df_dropDuplicates = df.drop_duplicates()    

df_dropDuplicates = df_dropDuplicates.reset_index(drop=True)

    
for i in df_dropDuplicates.index:
    if not (df_dropDuplicates.loc[i , 'ID'].startswith("MED",0,3)):
        df_dropDuplicates.drop(index=i,axis=0,inplace=True)   

df_dropDuplicates = df_dropDuplicates.reset_index(drop=True)    

df_dropDuplicates.to_excel(cd + '/dropDuplicates/med166_dropDuplicates.xlsx') 


