# -*- coding: utf-8 -*-


import pickle

def xgboost_shap(X,y):
    import xgboost
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    import shap


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


    skf = StratifiedKFold(n_splits=10, random_state=0)
    ## note X,y is dataFrame, not array, so .split(X,y) is not right
    t = y.values
    allAUC=[]
    allSHAP=[]
    for train_index, test_index in skf.split(np.zeros(len(t)),t):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        D_train = xgboost.DMatrix(X_train, label=y_train)
        D_test = xgboost.DMatrix(X_test, label=y_test)

        steps = 20  # The number of training iterations

        model = xgboost.train(param, D_train, steps)
    
        'weight'
        preds = model.predict(D_test)
        auc = roc_auc_score(y_test, preds)
        print('auc=', auc)    
        allAUC.append(auc)
    
        'score'
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        allSHAP.append(shap_values)
        print('shap_values is ok!')

    
    allAUC_allSHAP = {'allAUC': allAUC, 'allSHAP': allSHAP}   
    
    return allAUC_allSHAP 


if __name__== '__main__':
    filepath = '.../AKI_age/methods/'
    f=open(filepath + 'Data.pkl','rb')
    Data=pickle.load(f)
    f.close()
    
    for g in range(len(Data)):
        data = Data[g] 
        X = data.iloc[:,:-1]
        y = data['label']

        file=open(cd+'/xgboost_shap/group'+ str(g+1)+'_AUC_SHAP.pkl','wb')
        pickle.dump(allAUC_allSHAP,file)
        file.close()    

        print('===== group '+str(g+1)+' is over! =====')




