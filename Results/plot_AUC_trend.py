# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:44:37 2019

@author: wulj
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

#import os
import pandas as pd

cd = 'F:/AKI_age/'
f = open(cd+'/prediction_model_wulj/Data.pkl','rb') 
#os.listdir(".")
#os.listdir('prediction_model_wulj')
#f = open('prediction_model_wulj/Data.pkl','rb')
Data = pickle.load(f)
f.close()





def auc_trend(g):

    data = Data[g] 
    X = data.iloc[:,:-1]
    #X.head()
    y = data['label']
    t = y.values


    #f = open(cd+'/shap_importance_wulj/weight_shap/group'+str(g+1)+'_stable_shap.pkl','rb')运行出错找不到文件，用下面替代
    #os.listdir(".")
    #os.listdir('shap_importance_wulj')
    f1 = open(cd+'/prediction_model_wulj/weight_shap/group'+str(g+1)+'_stable_shap.pkl','rb')
    data = pickle.load(f1)
    f1.close()

    weight_shap = data['weight_average_shap']
    df_weight = pd.DataFrame(weight_shap,columns=X.columns)
    temp = df_weight.loc[:,~ (df_weight==0).all()]
    temp = np.abs(temp).mean(0)
    ind = temp.sort_values(ascending=False)
    
    param = {
        # can be gbtree or gblinear
        'booster':'gbtree', 
        # choose logistic regression loss function for binary classification
        'objective':'binary:logistic',
        # step size shrinkage
        'eta': 0.3,
        # minimum loss reduction required to make a further partition
        'gamma': 0.25,
        # minimum sum of instance weight(hessian) needed in a child
        'min_child_weight': 1,
        # maximum depth of a tree
        'max_depth': 6,
        # boosting learning rate
        'learning_rate': 0.3,
        # subsample ratio of the training instance
        #'subsample': 1
        }
    steps = 20  # The number of training iterations

    skf = StratifiedKFold(n_splits=10, random_state=0) 
    
    ##########################################
    big_auc=[]    
    indx = np.arange(50,450,50)
    for i in range(len(indx)):
        temp = indx[i]
        temp1 = ind[:temp].index
        X0 = X.loc[:,temp1]
        
        
        allAUC=[]       
        for train_index, test_index in skf.split(np.zeros(len(t)),t):
            X_train, X_test = X0.iloc[train_index,:], X0.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            D_train = xgboost.DMatrix(X_train, label=y_train)
            D_test = xgboost.DMatrix(X_test, label=y_test)
            
            model = xgboost.train(param, D_train, steps)
        
            preds = model.predict(D_test)
            auc = roc_auc_score(y_test, preds)
            print('auc=', auc)    
            allAUC.append(auc)
        mAUC = np.mean(allAUC)
        print('XGBoost AUC=', mAUC)
        big_auc.append([mAUC,mAUC - np.percentile(allAUC,2.5)])

         
    
    return big_auc, ind

big_auc0, ind0 = auc_trend(0)
######################################################
big_auc1, ind1 = auc_trend(1)
big_auc2, ind2 = auc_trend(2)
big_auc3, ind3 = auc_trend(3)
#####################################################
def draw():
    fig, axes = plt.subplots(4,2,figsize=(10,20))
    #设置横纵坐标的名称以及对应字体格式
    font = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 18}
    ##############  axes[0,0]
    yy = ind0.values
    xx = np.arange(1,len(yy)+1,1)
    axes[0,0].bar(xx,yy)
    axes[0,0].set_xlabel('Top-k feature')
    axes[0,0].set_ylabel('Feature importance')
    axes[0,0].set_title('Group 1 (age 18-35)')
    
    ############# axes[0,1]
    xx = np.array([1,2,3,4,5,6,7,8])
    yy = np.array([h[0] for h in big_auc0])
    err = np.array([h[1] for h in big_auc0])
    
    axes[0,1].errorbar(xx,yy,yerr=err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
    axes[0,1].plot(xx,yy,'g')
    axes[0,1].set_xlabel('Top k features')
    axes[0,1].set_ylabel('AUC [95% CI]')
    axes[0,1].set_xticks([1,2,3,4,5,6,7,8])
    axes[0,1].set_xticklabels(['50','100','150','200','250','300','350','400'])
    axes[0,1].set_title('Group 1 (age 18-35)')
    ##############  axes[1,0]
    yy = ind1.values
    xx = np.arange(1,len(yy)+1,1)
    axes[1,0].bar(xx,yy)
    axes[1,0].set_xlabel('Top-k feature')
    axes[1,0].set_ylabel('Feature importance')
    axes[1,0].set_title('Group 2 (age 36-55)')    
    ############# axes[1,1]
    xx = np.array([1,2,3,4,5,6,7,8])
    yy = np.array([h[0] for h in big_auc1])
    err = np.array([h[1] for h in big_auc1])
    
    axes[1,1].errorbar(xx,yy,yerr=err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
    axes[1,1].plot(xx,yy,'g')
    axes[1,1].set_xlabel('Top k features')
    axes[1,1].set_ylabel('AUC [95% CI]')
    axes[1,1].set_xticks([1,2,3,4,5,6,7,8])
    axes[1,1].set_xticklabels(['50','100','150','200','250','300','350','400'])    
    axes[1,1].set_title('Group 2 (age 36-55)')        
    ##############  axes[2,0]
    yy = ind2.values
    xx = np.arange(1,len(yy)+1,1)
    axes[2,0].bar(xx,yy)
    axes[2,0].set_xlabel('Top-k feature')
    axes[2,0].set_ylabel('Feature importance')
    axes[2,0].set_title('Group 3 (age 56-65)')        
    ############# axes[2,1]
    xx = np.array([1,2,3,4,5,6,7,8])
    yy = np.array([h[0] for h in big_auc2])
    err = np.array([h[1] for h in big_auc2])
    
    axes[2,1].errorbar(xx,yy,yerr=err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
    axes[2,1].plot(xx,yy,'g')
    axes[2,1].set_xlabel('Top k features')
    axes[2,1].set_ylabel('AUC [95% CI]')
    axes[2,1].set_xticks([1,2,3,4,5,6,7,8])
    axes[2,1].set_xticklabels(['50','100','150','200','250','300','350','400'])
    axes[2,1].set_title('Group 3 (age 56-65)') 
    ##############  axes[3,0]
    yy = ind3.values
    xx = np.arange(1,len(yy)+1,1)
    axes[3,0].bar(xx,yy)
    axes[3,0].set_xlabel('Top-k feature')
    axes[3,0].set_ylabel('Feature importance')
    axes[3,0].set_title('Group 4 (age >65)')     
    ############# axes[3,1]
    xx = np.array([1,2,3,4,5,6,7,8])
    yy = np.array([h[0] for h in big_auc3])
    err = np.array([h[1] for h in big_auc3])
    
    axes[3,1].errorbar(xx,yy,yerr=err,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
    axes[3,1].plot(xx,yy,'g')
    axes[3,1].set_xlabel('Top k features')
    axes[3,1].set_ylabel('AUC [95% CI]')
    axes[3,1].set_xticks([1,2,3,4,5,6,7,8])
    axes[3,1].set_xticklabels(['50','100','150','200','250','300','350','400'])
    axes[3,1].set_title('Group 4 (age >65)') 
    
    plt.savefig(cd + '/prediction_model_wulj/AUC_trend_plot/'+'AUC trend for 4 groups.png',dpi=400,bbox_inches='tight')
    plt.savefig(cd + '/prediction_model_wulj/AUC_trend_plot/'+'AUC trend for 4 groups.pdf',dpi=400,bbox_inches='tight')
    plt.show()

d = draw()





