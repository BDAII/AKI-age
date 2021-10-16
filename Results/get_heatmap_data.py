# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns

cd = '.../AKI_age/shap_importance/'

f = open(cd+'final_data_wshap.pkl','rb')
Data= pickle.load(f)
f.close()

def get_heatmap(Data):
    data = Data['data']
    wshap = Data['wshap']

    unionID = pd.read_excel('.../AKI_age/realFactors/unionF/unionID.xlsx',index_col=0)
    finID = list(unionID['unionF'])
    ############### map name ###############
    csv_map = pd.read_csv(cd+'AKI_age_map.csv')

    del csv_map['Index']
    dict_name = csv_map.set_index('ID').T.to_dict('list')


    #################################################################
    def get_direction(ID,data0,wshap0):    
        temp = data0.loc[:,ID]
        temp1 = temp[temp>0]
        temp1 = list(temp1.index)    
        temp3 = wshap0.loc[temp1,ID] # or wshap0   
        pos_effect = np.mean(temp3)
    
        return pos_effect

    def get_wshap(g):
        data0 = data[g-1]
        wshap0 = wshap[g-1]
    
        finData = []
        for i in finID:
            try:
                pos_effect = get_direction(i,data0,wshap0)
            except:
                pos_effect = np.nan
            
            finData.append(pos_effect)
        
        return finData


    wshap1 = np.round(get_wshap(1),3)        
    wshap2 = np.round(get_wshap(2),3)         
    wshap3 = np.round(get_wshap(3),3)         
    wshap4 = np.round(get_wshap(4),3)         

    finName = [x+str(dict_name[x]) for x in finID]

    df = pd.DataFrame({'ID':finName,'18-35':wshap1,'36-55':wshap2,'56-65':wshap3,'>65':wshap4})       
        
    df_heatmap = df.iloc[:,1:].copy().set_index(df.loc[:,'ID'])        
        
    # save
    df_heatmap.to_excel('.../AKI_age/realFactors/unionF/df_heatmap.xlsx')

    return df_heatmap
        
        
## ==================figure =======================
def draw_heatmap(df_heatmap):
    fig = plt.figure(figsize=(12,38))

    sns_plot = sns.heatmap(df_heatmap,annot=True,cmap='Oranges',annot_kws={'size':25})
    # annot_kws={'size':25,'weight':'bold', 'color':'black'}

    sns_plot.tick_params(labelsize=30)

    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=25) 

    plt.ylabel('')
    plt.title('Heatmap for mean(weighted SHAP values)',fontsize=35)
    fig.savefig('.../AKI_age/realFactors/unionF/heatmap.png',bbox_inches='tight')
    plt.show()        
 

 
df_heatmap = get_heatmap(Data)        
draw_heatmap(df_heatmap)        
        
        
        
        
        
        
        
        
        
        







