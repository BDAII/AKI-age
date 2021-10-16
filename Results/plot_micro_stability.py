# -*- coding: utf-8 -*-


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cd = '.../AKI_age/compute_stability/'

g=0 # age group

print('=============== age group =', g+1,'========')
Data = []

for i in range(10):
    f = open(cd +'/data_forStability/group'+str(g+1)+'/'+str(i+1)+'/FinalData.pkl','rb')
    data = pickle.load(f)
    Data.append(data)

del data, i
####################################
delt = 1e-6


def get_cos_similarity(a,b):
    mya = np.array(a)
    myb = np.array(b)
    cos1 = np.sum(mya*myb)
    cos21 = np.sqrt(np.sum(mya*mya))
    cos22 = np.sqrt(np.sum(myb*myb))
    result = cos1/float(cos21*cos22)
    return result


def get_cossimi(X):
    s = X.shape[1]
    ret = np.zeros((s,s))
    for i in range(s):
        for j in range(s):
            temp0 = get_cos_similarity(X[:,i],X[:,j])
            ret[i,j] = temp0
    return ret
        
    

def weight_feature_shap():
    all_stability = [[]]
    temp0 = Data[0]['weight_average_shap']
    sN = temp0.shape[0]
    fN = temp0.shape[1]
    for i in range(fN):
        feature_shap = [[]]*sN
        S = len(Data)
        for j in range(S):
            temp = Data[j]['weight_average_shap'].iloc[:,i]
            feature_shap = np.column_stack((feature_shap, temp.values + delt))
        
        feature_similarity = get_cossimi(feature_shap)
        #feature_similarity = np.corrcoef(feature_shap,rowvar=0)
        feature_stability = (feature_similarity.sum()-S*1)/2/(S*(S-1)/2)
        all_stability = np.column_stack((all_stability,feature_stability))
        
    result = pd.DataFrame(all_stability, columns=temp0.columns)
    return result
 
 
weight_feature_stability = weight_feature_shap()


#######################################
def mean_feature_shap():
    all_stability = [[]]
    temp0 = Data[0]['mean_shap']
    sN = temp0.shape[0]
    fN = temp0.shape[1]
    for i in range(fN):
        feature_shap = [[]]*sN
        S = len(Data)
        for j in range(S):
            temp = Data[j]['mean_shap'].iloc[:,i]
            feature_shap = np.column_stack((feature_shap, temp.values + delt))
            
        feature_similarity = get_cossimi(feature_shap)
        #feature_similarity = np.corrcoef(feature_shap,rowvar=0)
        feature_stability = (feature_similarity.sum()-S*1)/2/(S*(S-1)/2)
        all_stability = np.column_stack((all_stability,feature_stability))
        
    result = pd.DataFrame(all_stability, columns=temp0.columns)
    return result
   
   
mean_feature_stability = mean_feature_shap()

#######################################
def raw_feature_shap():
    all_stability = [[]]
    data = Data[0]['allSHAP'] # 只选第一个10-fold的结果计算稳定性
    sN = np.shape(data[0])[0]
    fN = np.shape(data[0])[1]
    S = len(data)
    for i in range(fN):
        feature_shap = [[]]*sN
        for j in range(S):
            temp = data[j].iloc[:,i]
            feature_shap = np.column_stack((feature_shap,temp.values + delt))
            
        feature_similarity = get_cossimi(feature_shap)
        #feature_similarity = np.corrcoef(feature_shap,rowvar=0)
        feature_stability = (feature_similarity.sum()-S*1)/2/(S*(S-1)/2)
        all_stability = np.column_stack((all_stability,feature_stability))
    
    result = pd.DataFrame(all_stability, columns=data[0].columns)
    return result
  
  
raw_feature_stability = raw_feature_shap()
    

########################################33
df = pd.concat([raw_feature_stability, mean_feature_stability, weight_feature_stability],axis=0,ignore_index=True)


for i in df.columns:
    if (df.loc[0,i]==df.loc[1,i]) & (df.loc[1,i]==df.loc[2,i]):
        del df[i]
        
for i in df.columns:
    if (np.round(df.loc[0,i],3)==1): 
        del df[i]
        


#####################################
def draw(df,d):
    fig, axes = plt.subplots(1,2,figsize=(15,4))
    x = np.array(range(df.shape[1]))
    
    axes[0].scatter(x,df.iloc[:1,:].values,c='b')
    axes[0].scatter(x,df.iloc[1,:].values,c='r')
    axes[1].scatter(x,df.iloc[2,:].values - df.iloc[1,:].values,c='g')
        
    axes[0].set_title('Group '+str(d)+': raw shap vs. weight shap')
    axes[1].set_title('Group '+str(d)+': weight shap vs. mean shap')
    
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Stability')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Difference in stability')
    
    plt.savefig(cd + 'result_stability/sample_nFeatures2/'+'Stability of shap values for group '
                + str(d)+'.png',dpi=400,bbox_inches='tight')
    
    plt.savefig(cd + 'result_stability/sample_nFeatures2/'+'Stability of shap values for group '
                + str(d)+'.pdf',dpi=400,bbox_inches='tight') 
    
    plt.show()   
 

d = draw(df,g+1)
  



















###########################################
#def draw(D1,D2,D3,d):
#    fig, axes = plt.subplots(1,3, figsize=(15,4))
#    
#    x = np.array(range(D1.shape[1]))
#
#    axes[0].scatter(x, D1.values)
#    axes[1].scatter(x, D2.values)
#    axes[2].scatter(x, D3.values)  
#     
#    axes[0].set_title('Group '+str(d)+': raw shap stability')    
#    axes[1].set_title('Group '+str(d)+': weight shap stability')
#    axes[2].set_title('Group '+str(d)+': mean shap stability') 
#    
#    axes[0].set_xlabel('Features')
#    axes[0].set_ylabel('Stability')
#    axes[1].set_xlabel('Features')
#    axes[1].set_ylabel('Stability')
#    axes[2].set_xlabel('Features')
#    axes[2].set_ylabel('Stability')
#    
#    plt.savefig(cd + 'result_stability/sample_nFeatures2/'+'Stability of shap values for group '
#                + str(d)+'.png',dpi=400,bbox_inches='tight')
#    
#    plt.show()
#
#d = draw(raw_feature_stability, mean_feature_stability, weight_feature_stability, g+1 )
    