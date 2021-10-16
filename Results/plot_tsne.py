# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import pickle
import numpy as np


cd = '.../AKI_age/'
f = open(cd+'/prediction_model/Data.pkl','rb') 
Data = pickle.load(f)
f.close()

'==============================================================='
def get_wSHAP_SHAP_rawX(g, n):
    data = Data[g] 
    X = data.iloc[:,:-1]
    y = data['label']


    f1 = open(cd+'/prediction_model_wulj/weight_shap/group'+str(g+1)+'_stable_shap.pkl','rb')
    data = pickle.load(f1)
    f1.close()
    
    f1 = open(cd+'/compute_stability/data_forStability/group'+str(g+1)+'/1/'+'FinalData.pkl','rb')
    data = pickle.load(f1)
    shapX = data['allSHAP'][0]
    f1.close()

    wshapX = data['weight_average_shap']
    df_weight = pd.DataFrame(wshapX,columns=X.columns)
    temp = df_weight.loc[:,~ (df_weight==0).all()]
    temp = np.abs(temp).mean(0)
    ind = temp.sort_values(ascending=False)
    
    temp1 = ind[:n].index  
    X0 = X.loc[:,temp1]
    shapX0 = shapX.loc[:,temp1]
    wshapX0 = wshapX.loc[:,temp1]
    
    return X0, y, ind, shapX0, wshapX0





#########################################
'use raw data to draw tsne'  
def data_tsne(g,n):
    X0, yy, ind, shapX0, wshapX0 = get_wSHAP_SHAP_rawX(g,n)
    model = TSNE(n_components=2,perplexity=5, init='pca')
    node_pos = model.fit_transform(X0)
    X0_tsne = pd.DataFrame(node_pos, index = X0.index)
    node_pos = model.fit_transform(shapX0)
    shapX0_tsne = pd.DataFrame(node_pos, index = shapX0.index)
    node_pos = model.fit_transform(wshapX0)
    wshapX0_tsne = pd.DataFrame(node_pos, index = wshapX0.index)

    return yy, X0_tsne, shapX0_tsne, wshapX0_tsne


g=0
n=50
yy, X0_tsne, shapX0_tsne, wshapX0_tsne = data_tsne(g,n)


def draw_tsne(g,n):
   
    #############
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1) 
    
    d=X0_tsne[yy==0]
    plt.plot(d[0],d[1],'b.',markersize=5,label = 'nonAKI')
    d=X0_tsne[yy==1]
    plt.plot(d[0],d[1],'r*',markersize=5,label = 'any AKI')
    plt.legend(loc='upper right')
    plt.xticks([]) 
    plt.yticks([]) 
    plt.title('A. t-SNE of raw data for nonAKI vs any AKI')
    #plt.show()

      
    plt.subplot(1, 2, 2) 
    
    d=wshapX0_tsne[yy==0]
    plt.plot(d[0],d[1],'b.',markersize=5,label = 'AKI 1')
    d=wshapX0_tsne[yy==1]
    plt.plot(d[0],d[1],'r*',markersize=5,label = 'AKI 2+3')
    plt.legend(loc='upper right')
    plt.xticks([])  
    plt.yticks([])  
    plt.title('B. t-SNE of weighted SHAP for nonAKI vs any AKI')
    #plt.show()
    
       
    plt.savefig(cd + '/accuracy/t-SNE/Group'+str(g+1)+'_tSNE.png',dpi=300,bbox_inches='tight')
    plt.savefig(cd + '/accuracy/t-SNE/Group'+str(g+1)+'_tSNE.eps',dpi=300,bbox_inches='tight')
    plt.savefig(cd + '/accuracy/t-SNE/Group'+str(g+1)+'_tSNE.pdf',dpi=300,bbox_inches='tight')
    plt.show()
    


##### run main 
draw_tsne()








