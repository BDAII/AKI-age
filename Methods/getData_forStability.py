# -*- coding: utf-8 -*-


import pickle
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import shap
import os
import pandas as pd



cd = '.../AKI_age/compute_stability/'
f=open(cd+'Data.pkl','rb')
Data=pickle.load(f)
f.close()




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

for g in [0,1,2,3]:    #range(len(Data)) [0,1,2,3]

    data = Data[g] 
    X = data.iloc[:,:-1]
    y = data['label']
    t = y.values
    
    
    path = cd +'/data_forStability/group'+str(g+1)
    if not os.path.exists(path):
        os.mkdir(path)
    
       
    for rs in range(10):  
        
        print('g = ', g+1,' and  rs = ', rs)
        file_path = cd +'/data_forStability/group'+str(g+1)+'/'+str(rs)
        if not os.path.exists(file_path):
            os.mkdir(file_path)
          
          
        D = pd.concat([X,y],axis=1)
        Dd = D.sample(frac=1)
        Xs = Dd.iloc[:,:-1]
        ys = Dd.iloc[:,-1:]
            
    
        skf = StratifiedKFold(n_splits=10, random_state = rs * 10) 
        allAUC=[]
        allSHAP=[]
       
       
        for train_index, test_index in skf.split(np.zeros(len(t)),t):
            X_train, X_test = Xs.iloc[train_index,:], Xs.iloc[test_index,:]
            y_train, y_test = ys.iloc[train_index,:], ys.iloc[test_index,:]
            D_train = xgboost.DMatrix(X_train, label=y_train)
            D_test = xgboost.DMatrix(X_test, label=y_test)
            
            model = xgboost.train(param, D_train, steps)
            
            'weight'
            preds = model.predict(D_test)
            auc = roc_auc_score(y_test, preds)
            allAUC.append(auc)
            
            'score'
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_values = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
            allSHAP.append(shap_values)
               
        
        print('======== group '+str(g+1)+' and rs= '+ str(rs)+'=============')
        weights = allAUC
        scores = allSHAP
        
        weight_average_shap = np.round(sum([weights[i]*scores[i] for i in range(len(weights))])/sum(weights),6)
        mean_shap = np.round(sum([scores[i] for i in range(len(scores))])/len(scores),6)
               

        FinalData = {'allAUC': allAUC, 'allSHAP': allSHAP, 'weight_average_shap':weight_average_shap,'mean_shap': mean_shap}
   
        f = open(file_path+'/FinalData.pkl','wb')
        pickle.dump(FinalData, f)
        f.close()
  

        


