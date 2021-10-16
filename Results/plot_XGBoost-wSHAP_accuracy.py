# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
#import os
import pandas as pd


cd = '.../AKI_age/'
f = open(cd+'/methods/Data.pkl','rb') 
Data = pickle.load(f)
f.close()


'==============================================================='
def get_wSHAP(g):
    data = Data[g] 
    X = data.iloc[:,:-1]
    #X.head()
    y = data['label']


    f1 = open(cd+'/methods/weight_shap/group'+str(g+1)+'_stable_shap.pkl','rb')
    data = pickle.load(f1)
    f1.close()

    weight_shap = data['weight_average_shap']
    df_weight = pd.DataFrame(weight_shap,columns=X.columns)
    temp = df_weight.loc[:,~ (df_weight==0).all()]
    temp = np.abs(temp).mean(0)
    ind = temp.sort_values(ascending=False)
    
    return y, ind, df_weight


def wSHAP_auc_trend(y, ind, df_weight,indx):
    
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
    all_auc=[]    
    
    for i in range(len(indx)):
        temp = indx[i]
        temp1 = ind[:temp].index       
        #X0 = X.loc[:,temp1]
        X0 = df_weight.loc[:,temp1]
        
        allAUC=[]       
        for train_index, test_index in skf.split(np.zeros(len(y.values)),y.values):
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
        print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])
        all_auc.append([mAUC,np.percentile(allAUC,2.5),np.percentile(allAUC,97.5),allAUC])
                
    return all_auc


'===================================================='

def get_SHAP(g):
    data = Data[g] 
    X = data.iloc[:,:-1]
    #X.head()
    y = data['label']
    

    f1 = open(cd+'/compute_stability/data_forStability/group'+str(g+1)+'/1/'+'FinalData.pkl','rb')
    data = pickle.load(f1)
    allSHAP = data['allSHAP']
    f1.close()
    
    return X,y, allSHAP


def SHAP_auc_trend(X,y, allSHAP,indx):

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
    all_auc=[]    
    
    for i in range(len(indx)):
        allAUC = []
        for k in range(len(allSHAP)):
            weight_shap = allSHAP[k]
            df_weight = pd.DataFrame(weight_shap,columns=X.columns)
            temp = df_weight.loc[:,~ (df_weight==0).all()]
            temp = np.abs(temp).mean(0)
            ind = temp.sort_values(ascending=False)
            
            temp = indx[i]
            temp1 = ind[:temp].index       
            #X0 = X.loc[:,temp1]
            X0 = df_weight.loc[:,temp1]
        
            k_allAUC=[]       
            for train_index, test_index in skf.split(np.zeros(len(y.values)),y.values):
                X_train, X_test = X0.iloc[train_index,:], X0.iloc[test_index,:]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                D_train = xgboost.DMatrix(X_train, label=y_train)
                D_test = xgboost.DMatrix(X_test, label=y_test)
            
                model = xgboost.train(param, D_train, steps)
        
                preds = model.predict(D_test)
                auc = roc_auc_score(y_test, preds)
                print('auc=', auc)    
                k_allAUC.append(auc)
            allAUC.extend(k_allAUC)
            
        mAUC = np.mean(allAUC)
        print('XGBoost AUC=', mAUC)
        print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])
        all_auc.append([mAUC,np.percentile(allAUC,2.5),np.percentile(allAUC,97.5),allAUC])                
    
    return all_auc

'====================================================================================='
def get_raw(g):
    data = Data[g] 
    X = data.iloc[:,:-1]
    y = data['label']


    f1 = open(cd+'/methods/weight_shap/group'+str(g+1)+'_stable_shap.pkl','rb')
    data = pickle.load(f1)
    f1.close()

    weight_shap = data['weight_average_shap']
    df_weight = pd.DataFrame(weight_shap,columns=X.columns)
    temp = df_weight.loc[:,~ (df_weight==0).all()]
    temp = np.abs(temp).mean(0)
    ind = temp.sort_values(ascending=False)
    
    return X, y, ind, df_weight


def raw_auc_trend(X, y, ind, df_weight,indx):
    
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
    all_auc=[]    
    
    for i in range(len(indx)):
        temp = indx[i]
        temp1 = ind[:temp].index       
        X0 = X.loc[:,temp1]
        #X0 = df_weight.loc[:,temp1]
        
        allAUC=[]       
        for train_index, test_index in skf.split(np.zeros(len(y.values)),y.values):
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
        print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])
        all_auc.append([mAUC,np.percentile(allAUC,2.5),np.percentile(allAUC,97.5),allAUC])
                
    return all_auc
        
        
        

'================================================='
def main_function():
    indx = np.arange(10,210,10)
    
    wshapAUC=[]
    shapAUC=[]
    rawAUC=[]
    
    for g in range(4):
        y, ind, df_weight = get_wSHAP(g)
        wshap_auc = wSHAP_auc_trend(y, ind, df_weight,indx)
        wshapAUC.append(wshap_auc)
        # for XGBoost-SHAP
        X,y, allSHAP = get_SHAP(g)
        shap_auc = SHAP_auc_trend(X,y, allSHAP,indx)
        shapAUC.append(shap_auc)
        # for raw XGBoost
        X,y, ind, df_weight = get_raw(g)
        raw_auc = raw_auc_trend(X,y, ind, df_weight,indx)
        rawAUC.append(raw_auc)

        print('===== group '+str(g+1)+' is over! =====')

    # to save
    wshapAUC_shapAUC_rawAUC = {'wshapAUC':wshapAUC,'shapAUC':shapAUC,'rawAUC':rawAUC}
    file=open(cd+'/methods/XGBoost-wSHAP/wshapAUC_shapAUC_rawAUC.pkl','wb')
    pickle.dump(wshapAUC_shapAUC_rawAUC,file)
    file.close()   

    return wshapAUC, shapAUC, rawAUC


wshapAUC, shapAUC, rawAUC = main_function()

#####################################################
import numpy as np
import matplotlib.pyplot as plt

def draw():
    indx = np.arange(10,210,10)
    
    fig, axes = plt.subplots(2,2,figsize=(11,10))

    ##############  axes[0,0]
    xx = indx
    y1_0 = np.array([h[0] for h in wshapAUC[0]])
    y1_1 = np.array([h[1] for h in wshapAUC[0]])
    y1_2 = np.array([h[2] for h in wshapAUC[0]])
    
    y2_0 = np.array([h[0] for h in shapAUC[0]])
    y2_1 = np.array([h[1] for h in shapAUC[0]])
    y2_2 = np.array([h[2] for h in shapAUC[0]])
    
    y3_0 = np.array([h[0] for h in rawAUC[0]])
    y3_1 = np.array([h[1] for h in rawAUC[0]])
    y3_2 = np.array([h[2] for h in rawAUC[0]])
    
    

    axes[0,0].plot(xx,y1_0,'r-*',label='Weighted SHAP values')
    axes[0,0].fill_between(xx, y1_1, y1_2,alpha=0.1,color='r')
    
    axes[0,0].plot(xx,y2_0,'b-s',label='Raw SHAP values')
    axes[0,0].fill_between(xx, y2_1, y2_2,alpha=0.1,color='b')
    
    axes[0,0].plot(xx,y3_0,'g-o',label='Original input samples')
    axes[0,0].fill_between(xx, y3_1, y3_2,alpha=0.1,color='g')
    
    axes[0,0].set_xlabel('Top k features')
    axes[0,0].set_ylabel('AUC (95% CI)')
    axes[0,0].set_xticks([10,25,50,75,100,125,150,175,200])
    axes[0,0].set_xticklabels(['10','25','50','75','100','125','150','175','200'])
    axes[0,0].set_title('Age group 1 (18-35)')
    
    ############# axes[0,1]
    xx = indx
    y1_0 = np.array([h[0] for h in wshapAUC[1]])
    y1_1 = np.array([h[1] for h in wshapAUC[1]])
    y1_2 = np.array([h[2] for h in wshapAUC[1]])
    
    y2_0 = np.array([h[0] for h in shapAUC[1]])
    y2_1 = np.array([h[1] for h in shapAUC[1]])
    y2_2 = np.array([h[2] for h in shapAUC[1]])
    
    y3_0 = np.array([h[0] for h in rawAUC[1]])
    y3_1 = np.array([h[1] for h in rawAUC[1]])
    y3_2 = np.array([h[2] for h in rawAUC[1]])
    
    

    axes[0,1].plot(xx,y1_0,'r-*',label='Weighted SHAP values')
    axes[0,1].fill_between(xx, y1_1, y1_2,alpha=0.1,color='r')
    
    axes[0,1].plot(xx,y2_0,'b-s',label='Raw SHAP values')
    axes[0,1].fill_between(xx, y2_1, y2_2,alpha=0.1,color='b')
    
    axes[0,1].plot(xx,y3_0,'g-o',label='Original input samples')
    axes[0,1].fill_between(xx, y3_1, y3_2,alpha=0.1,color='g')
    
    axes[0,1].set_xlabel('Top k features')
    axes[0,1].set_ylabel('AUC (95% CI)')
    axes[0,1].set_xticks([10,25,50,75,100,125,150,175,200])
    axes[0,1].set_xticklabels(['10','25','50','75','100','125','150','175','200'])
    axes[0,1].set_title('Age group 2 (36-55)')    

    
    ############# axes[1,0]
    xx = indx
    y1_0 = np.array([h[0] for h in wshapAUC[2]])
    y1_1 = np.array([h[1] for h in wshapAUC[2]])
    y1_2 = np.array([h[2] for h in wshapAUC[2]])
    
    y2_0 = np.array([h[0] for h in shapAUC[2]])
    y2_1 = np.array([h[1] for h in shapAUC[2]])
    y2_2 = np.array([h[2] for h in shapAUC[2]])
    
    y3_0 = np.array([h[0] for h in rawAUC[2]])
    y3_1 = np.array([h[1] for h in rawAUC[2]])
    y3_2 = np.array([h[2] for h in rawAUC[2]])
    
    

    axes[1,0].plot(xx,y1_0,'r-*',label='Weighted SHAP values')
    axes[1,0].fill_between(xx, y1_1, y1_2,alpha=0.1,color='r')
    
    axes[1,0].plot(xx,y2_0,'b-s',label='Raw SHAP values')
    axes[1,0].fill_between(xx, y2_1, y2_2,alpha=0.1,color='b')
    
    axes[1,0].plot(xx,y3_0,'g-o',label='Original input samples')
    axes[1,0].fill_between(xx, y3_1, y3_2,alpha=0.1,color='g')
    
    axes[1,0].set_xlabel('Top k features')
    axes[1,0].set_ylabel('AUC (95% CI)')
    axes[1,0].set_xticks([10,25,50,75,100,125,150,175,200])
    axes[1,0].set_xticklabels(['10','25','50','75','100','125','150','175','200'])
    axes[1,0].set_title('Age group 3 (56-65)')  

    
    ################## axes[1,1]    
    xx = indx
    y1_0 = np.array([h[0] for h in wshapAUC[3]])
    y1_1 = np.array([h[1] for h in wshapAUC[3]])
    y1_2 = np.array([h[2] for h in wshapAUC[3]])
    
    y2_0 = np.array([h[0] for h in shapAUC[3]])
    y2_1 = np.array([h[1] for h in shapAUC[3]])
    y2_2 = np.array([h[2] for h in shapAUC[3]])
    
    y3_0 = np.array([h[0] for h in rawAUC[3]])
    y3_1 = np.array([h[1] for h in rawAUC[3]])
    y3_2 = np.array([h[2] for h in rawAUC[3]])
    
    

    axes[1,1].plot(xx,y1_0,'r-*',label='Weighted SHAP values')
    axes[1,1].fill_between(xx, y1_1, y1_2,alpha=0.1,color='r')
    
    axes[1,1].plot(xx,y2_0,'b-s',label='Raw SHAP values')
    axes[1,1].fill_between(xx, y2_1, y2_2,alpha=0.1,color='b')
    
    axes[1,1].plot(xx,y3_0,'g-o',label='Original input samples')
    axes[1,1].fill_between(xx, y3_1, y3_2,alpha=0.1,color='g')
    
    axes[1,1].set_xlabel('Top k features')
    axes[1,1].set_ylabel('AUC (95% CI)')
    axes[1,1].set_xticks([10,25,50,75,100,125,150,175,200])
    axes[1,1].set_xticklabels(['10','25','50','75','100','125','150','175','200'])
    axes[1,1].set_title('Age group 4 (>65)')  
    
    plt.legend(loc='lower right')
    
    plt.savefig(cd + '/methods/accuracy/'+'XGBoost_wSHAP_AUC trend.png',dpi=300,bbox_inches='tight')
    plt.savefig(cd + '/methods/accuracy/'+'XGBoost_wSHAP_AUC trend.eps',dpi=300,bbox_inches='tight')
    plt.savefig(cd + '/methods/accuracy/'+'XGBoost_wSHAP_AUC trend.pdf',dpi=300,bbox_inches='tight')
    plt.savefig(cd + '/methods/accuracy/'+'XGBoost_wSHAP_AUC trend.tiff',dpi=300,bbox_inches='tight')
    plt.show()


d = draw()
