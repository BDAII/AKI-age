# -*- coding: utf-8 -*-

import pickle
import numpy as np

cd = '.../AKI_age/xgboost_shap/'


for g in range(4):
    
    f = open(cd+'/group'+ str(g+1)+'_AUC_SHAP.pkl','rb')
    data = pickle.load(f)
    f.close()
    
    weights = data['allAUC']
    scores = data['allSHAP']
    
    weight_average_shap = np.round(sum([weights[i]*scores[i] for i in range(len(weights))])/sum(weights),6)
    mean_shap = np.round(sum([scores[i] for i in range(len(scores))])/len(scores),6)
    
    stable_shap = {'weight_average_shap':weight_average_shap,'mean_shap': mean_shap}
    f = open(cd+'/weight_shap/group'+str(g+1)+'_stable_shap.pkl','wb')
    pickle.dump(stable_shap,f)
    f.close()




