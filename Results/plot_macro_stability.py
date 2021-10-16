# -*- coding: utf-8 -*-


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

######################
cd = '.../AKI_age/compute_stability/'
g=3 # age group
print('=============== age group =', g+1,'========')
Data = []

for i in range(10):
    f = open(cd +'/data_forStability/group'+str(g+1)+'/'+str(i+1)+'/FinalData.pkl','rb')
    data = pickle.load(f)
    Data.append(data)
    
feature_name = data['weight_average_shap'].columns


def weight_shap():
    all_weight =pd.Series(index=data['weight_average_shap'].columns) # nan series
    for i in range(len(Data)):
        temp0 = Data[i]['weight_average_shap']
        temp1 = np.abs(temp0).mean(0)
        all_weight = pd.concat([all_weight,temp1],axis=1)
        
    temp3 = all_weight.values[:,1:] # delete the first nan column
     
    result = np.corrcoef(temp3, rowvar=0)
    return result
 
 
 
def mean_shap():
    all_weight =pd.Series(index=data['mean_shap'].columns) # nan series
    for i in range(len(Data)):
        temp0 = Data[i]['mean_shap']
        temp1 = np.abs(temp0).mean(0)
        all_weight = pd.concat([all_weight,temp1],axis=1)
        
    temp3 = all_weight.values[:,1:] # delete the first nan column
     
    result = np.corrcoef(temp3, rowvar=0)
    return result    
 
     
def raw_shap():
    # since 10-fold cv, we only choose  1 results  i.e., 10 CV raw shap values
    raw_shap = pd.Series(index=Data[0]['allSHAP'][0].columns)
    temp0 = Data[0]['allSHAP']
    for i in range(10):
        temp1 = np.abs(temp0[i]).mean(0)
        raw_shap = pd.concat([raw_shap, temp1], axis=1)
    
    temp2 = raw_shap.values[:,1:]
    result = np.corrcoef(temp2, rowvar=0)
    return result
   

weight_shap = weight_shap()
mean_shap = mean_shap()              
raw_shap = raw_shap()


########### save data ===========
resultData = {'weight_shap':weight_shap,'mean_shap':mean_shap,'raw_shap':raw_shap}
f = open( cd + 'result_stability/feature_summary1/group'+str(g)+'_similarity.pkl','wb')
pickle.dump(resultData,f)
f.close()

    
def draw(D1,D2,D3,d):
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

 
    sns.heatmap(D1,annot=True,cmap='cool',ax=ax1)
    sns.heatmap(D2,annot=True,cmap='cool',ax=ax2)
    sns.heatmap(D3,annot=True,cmap='cool',ax=ax3)
    

    temp0 = list(range(10))
    xLabel = [str(x+1) for x in temp0]
    yLabel = xLabel
    
 
    ax1.set_yticks(np.arange(0.5,10.5,1))    
    ax1.set_yticklabels(yLabel)    
    ax1.set_xticks(np.arange(0.5,10.5,1))    
    ax1.set_xticklabels(xLabel)  


    
    ax2.set_yticks(np.arange(0.5,10.5,1))
    ax2.set_yticklabels(yLabel)
    ax2.set_xticks(np.arange(0.5,10.5,1))
    ax2.set_xticklabels(xLabel) 

    ax3.set_yticks(np.arange(0.5,10.5,1))
    ax3.set_yticklabels(yLabel)
    ax3.set_xticks(np.arange(0.5,10.5,1))
    ax3.set_xticklabels(xLabel)    
       
    ax1.set_title('Group '+str(d)+': mean(|raw_shap|)')    
    ax3.set_title('Group '+str(d)+': mean(|weight_shap|)')
    ax2.set_title('Group '+str(d)+': mean(|mean_shap|)')    
    
    plt.savefig(cd + 'result_stability/feature_summary1/'+'heatmap of similarity for group '
                + str(d)+'.png',dpi=400,bbox_inches='tight')
    
    plt.savefig(cd + 'result_stability/feature_summary1/'+'heatmap of similarity for group '
                + str(d)+'.pdf',dpi=400,bbox_inches='tight')
        
    plt.show()

d = draw(raw_shap, mean_shap,weight_shap, g+1 )
    
