# -*- coding: utf-8 -*-


import pandas as pd
import pickle
import numpy as np

cd = '.../AKI_age/compute_stability/'

final_weightSHAP = []
for i in [1,2,3,4]:
    f = open(cd+'data_forStability/group'+ str(i)+'/FinalData.pkl','rb')
    temp0 = pickle.load(f)
    temp1 = temp0['weight_average_shap']
    final_weightSHAP.append(temp1)
    

file=open(cd+'final_weightSHAP.pkl','wb')
pickle.dump(final_weightSHAP,file)
file.close()


