# -*- coding: utf-8 -*-


import pickle
import pandas as pd
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
def topk_name(g,k):
    data0 = data[g]
    wshap0 = wshap[g]

    temp = np.abs(wshap0).mean(0)
    ind = temp.sort_values(ascending=False)

    temp1 = ind[:k].index
    df_wshap = wshap0.loc[:,temp1]

    col_ID = list(df_wshap.columns)
    col_name = [str(dict_name[x]) for x in col_ID]

    col_rank = list(np.arange(1,151,1))


    df_Rank = pd.DataFrame(col_rank,columns=['Rank'])
    df_ID = pd.DataFrame(col_ID,columns=['ID'])
    df_Name = pd.DataFrame(col_name,columns=['Name'])
    df = pd.concat([df_ID,df_Name,df_Rank,],axis=1)
    df.to_excel(cd+'/get_top150_name/top150_group'+str(g)+'.xlsx',index = False)


k=150

for g in range(4):
    topk_name(g,k)