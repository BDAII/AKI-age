# -*- coding: utf-8 -*-

import pickle
import random


################################################################################
#'=======================Model: XGBoost========================================='
################################################################################
def Xgboost(X,y):
    import xgboost
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

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
    for train_index, test_index in skf.split(np.zeros(len(t)),t): 
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        D_train = xgboost.DMatrix(X_train, label=y_train)
        D_test = xgboost.DMatrix(X_test, label=y_test)

        steps = 20  # The number of training iterations

        model = xgboost.train(param, D_train, steps)
        preds = model.predict(D_test)
        auc = roc_auc_score(y_test, preds)
        print('auc=', auc)
        allAUC.append(auc)

    AUC = np.mean(allAUC)
    print('XGBoost AUC=', AUC)
    print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])

    return allAUC
    
    
################################################################################
#'=======================Model: GBM========================================='
################################################################################
def GBM(X,y):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=10, random_state=0)
      
    #clf = GradientBoostingClassifier(random_state=10)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,subsample=0.8,loss='deviance',max_features='sqrt',
                                    max_depth=3,min_samples_split=10,min_samples_leaf=3,min_weight_fraction_leaf=0,
                                    random_state=10)
    ## note X,y is dataFrame, not array, so .split(X,y) is not right             
    t = y.values
    allAUC=[]
    for train_index, test_index in skf.split(np.zeros(len(t)),t):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        predicted = probs[:,1]
        auc = roc_auc_score(y_test, predicted)
    
        print('auc=', auc)
        allAUC.append(auc)

    AUC = np.mean(allAUC)
    print('LightGBM AUC=', AUC)
    print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])

    return allAUC
    
    
################################################################################
#'=======================Model: Random Forest==================================='
################################################################################
def RF(X,y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=10, random_state=0)

    clf = RandomForestClassifier(n_estimators=300, random_state=10,n_jobs=-1)
    ## note X,y is dataFrame, not array, so .split(X,y) is not right             
    t = y.values
    allAUC=[]
    for train_index, test_index in skf.split(np.zeros(len(t)),t):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        predicted = probs[:,1]
        auc = roc_auc_score(y_test, predicted)
    
        print('auc=', auc)
        allAUC.append(auc)

    AUC = np.mean(allAUC)
    print('Random Forest AUC=', AUC)
    print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])

    return allAUC
    
    
################################################################################
#'=======================Model: LinearSVC =================================='
################################################################################
def linearSVC(X,y):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=10, random_state=0)

    clf = LinearSVC(penalty='l2', loss='squared_hinge', 
                    dual=False, tol=0.0001, C=2.0, multi_class='ovr', 
                    fit_intercept=True, intercept_scaling=1, class_weight='balanced', 
                    verbose=0, random_state=0, max_iter=100)

    ## note X,y is dataFrame, not array, so .split(X,y) is not right             
    t = y.values
    allAUC=[]
    for train_index, test_index in skf.split(np.zeros(len(t)),t):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
    
        '''
        since 'LinearSVC' object has no attribute 'predict_proba'
        we use its decision function (predict confidence scores for samples)
        '''
        prob_pos = clf.decision_function(X_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        auc = roc_auc_score(y_test, prob_pos)
    
        print('auc=', auc)
        allAUC.append(auc)

    AUC = np.mean(allAUC)
    print('Linear SVC AUC=', AUC)
    print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])

    return allAUC
    
    
################################################################################
#'=======================Model: Logistic Regression ======================='
################################################################################
def LR(X,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=10, random_state=0)

    clf = LogisticRegression()

    ## note X,y is dataFrame, not array, so .split(X,y) is not right             
    t = y.values
    allAUC=[]
    
    for train_index, test_index in skf.split(np.zeros(len(t)),t):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        predicted = probs[:,1]
        auc = roc_auc_score(y_test, predicted)
    
        print('auc=', auc)
        allAUC.append(auc)

    AUC = np.mean(allAUC)
    print('Logistic Regression AUC=', AUC)
    print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])

    return allAUC
    
    
################################################################################
#'=======================Model: Naive Bayes ==========================='
################################################################################
def NB(X,y):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=10, random_state=0)

    clf = BernoulliNB()

    ## note X,y is dataFrame, not array, so .split(X,y) is not right             
    t = y.values
    allAUC=[]
    for train_index, test_index in skf.split(np.zeros(len(t)),t):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        predicted = probs[:,1]
        auc = roc_auc_score(y_test, predicted)
    
        print('auc=', auc)
        allAUC.append(auc)

    AUC = np.mean(allAUC)
    print('Naive Bayes AUC=', AUC)
    print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])

    return allAUC
    
    
################################################################################
#'=======================Model: Decision Trees ==========================='
################################################################################
def DT(X,y):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=10, random_state=0)

    clf = DecisionTreeClassifier()

    ## note X,y is dataFrame, not array, so .split(X,y) is not right             
    t = y.values
    allAUC=[]
    for train_index, test_index in skf.split(np.zeros(len(t)),t):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        predicted = probs[:,1]
        auc = roc_auc_score(y_test, predicted)
    
        print('auc=', auc)
        allAUC.append(auc)

    AUC = np.mean(allAUC)
    print('Decision Trees AUC=', AUC)
    print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])

    return allAUC
    
    
################################################################################
#'=======================Model: Neural network ==========================='
################################################################################
def NN(X,y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=10, random_state=0)

    clf = MLPClassifier(hidden_layer_sizes=(300,100,50))
    ## note X,y is dataFrame, not array, so .split(X,y) is not right             
    t = y.values
    allAUC=[]
    for train_index, test_index in skf.split(np.zeros(len(t)),t):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        predicted = probs[:,1]
        auc = roc_auc_score(y_test, predicted)
    
        print('auc=', auc)
        allAUC.append(auc)

    AUC = np.mean(allAUC)
    print('Neural Network AUC=', AUC)
    print('%95CI=',[ np.percentile(allAUC,2.5), np.percentile(allAUC,97.5)])

    return allAUC
    
 
def get_auc(group, method):
    data = Data[g] 

    index = [i for i in range(len(data))] 
    random.shuffle(index)
    data = data.iloc[index,:]

    X = data.iloc[:,:-1]
    y = data['label']

    if method == 'XGBoost':
        allAUC = Xgboost(X,y)
    elif method == 'GBM':
        allAUC = GBM(X,y)
    elif method == 'RandomForest':
        allAUC = RF(X,y)
    elif method == 'linearSVC':
        allAUC = linearSVC(X,y)
    elif method == 'LogisticRegression':
        allAUC = LR(X,y)
    elif method == 'NaiveBayes':
        allAUC = NB(X,y)
    elif method == 'DecisionTrees':
        allAUC = DT(X,y)
    elif method == 'NeuralNetwork'
        allAUC = NN(X,y)
    else:
        print('Error input: method!')
        
    return allAUC
 
#################################################################
if __name__== '__main__':
    cd = '.../AKI_age/methods/'
    f=open(cd+'Data.pkl','rb')
    Data=pickle.load(f)
    f.close()
    methods = ['XGBoost','GBM','RandomForest','linearSVC','LogisticRegression','NaiveBayes','DecisionTrees','NeuralNetwork']
    for g in range(len(Data)):
        for m in range(len(methods)):
            allAUC = get_auc(g, methods[m])
        
   