# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt

cd = '.../AKI_age/shap_importance/med_mining/'

rawdata = pd.read_csv(cd+'med166_shap_direction.csv')

rawdata['G1'] = rawdata.loc[:,'g1_1'] - rawdata.loc[:,'g1_0'] 
rawdata['G2'] = rawdata.loc[:,'g2_1'] - rawdata.loc[:,'g2_0'] 
rawdata['G3'] = rawdata.loc[:,'g3_1'] - rawdata.loc[:,'g3_0'] 
rawdata['G4'] = rawdata.loc[:,'g4_1'] - rawdata.loc[:,'g4_0'] 


data = rawdata.loc[:,['medID','medName','G1','G2','G3','G4']]


fig = plt.figure(figsize=(8,4))

plt.ylabel('mean(weight_shap(if med>0))-mean(weight_shap(if med=0))')
plt.xlabel('166 medications')
plt.ylim([-0.4,1.8])
plt.xlim([0,170])

x = list(range(1,167,1))
yname = data['medID']+data['medName']

plt.scatter(x,data.loc[:,'G1'],label='G1: 18-35',marker='s' , cmap= 'mediumslateblue',alpha=1)
plt.scatter(x,data.loc[:,'G2'],label='G2: 36-55',marker='o' , cmap= 'lawngreen',alpha=1)
plt.scatter(x,data.loc[:,'G3'],label='G3: 56-65',marker='^' , cmap= 'gold',alpha=1)
plt.scatter(x,data.loc[:,'G4'],label='G4:  >65',marker='p' , cmap= 'magenta',alpha=1)

plt.title('Relative risk of drug factors')
plt.legend()
plt.savefig(cd+'med166_scatter_plot.png',bbox_inches='tight',dpi=300)

plt.show()




