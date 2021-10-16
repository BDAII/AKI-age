# -*- coding: utf-8 -*-



import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity



cd = '.../AKI_age/shap_importance/'

#============ input data ===============
f = open(cd+'final_data_wshap.pkl','rb')
Data = pickle.load(f)
f.close()

data = Data['data']
wshap = Data['wshap']

############### map name ###############
csv_map = pd.read_csv(cd+'AKI_age_map.csv')

del csv_map['Index']
dict_name = csv_map.set_index('ID').T.to_dict('list')

#==============================================
#'Use cos similarity to calculate similarity and sort for the later knowledge network'

def get_cosSim(x,y):
    myx=np.array(x)
    myy=np.array(y)
    cos1=np.sum(myx*myy)
    cos21=np.sqrt(sum(myy*myy))
    cos22=np.sqrt(sum(myx*myx))
    temp = cos1/float(cos22*cos21)
    return temp


def get_correlation(g,ID):
    data_g = data[g-1]
    
    result_cosSim = pd.Series(index=data_g.columns)
    for i in data_g.columns:
        temp = get_cosSim(data_g.loc[:,i], data_g.loc[:,ID])
        result_cosSim[i] = temp
        
    finalD = result_cosSim.sort_values(ascending=False)[1:]
    return finalD

#===================================
def compute_cosCorrM(g,iD,k):
    data_g = data[g-1]
    
    file_path = cd + '/med_knowledge/plot_networkx/' +iD
    if not os.path.exists(file_path):
        os.mkdir(file_path)
        
    ############## 
    cosSim = get_correlation(g,iD)
    temp = list(cosSim.index)
    name = [str(dict_name[x]) for x in temp]
    cosSimA = pd.DataFrame({'ID':cosSim.index,'name':name,'cosSim':cosSim.values})
    cosSimA.to_csv(cd+'/med_knowledge/plot_networkx/'+iD+'/G'+str(g)+'_'+iD+'_cosSim.csv',float_format='%.4f')
    
    ############
    temp0 = cosSimA.iloc[:k,:]
    cosName = temp0['ID']
    
    ###########
    tempData = data_g.loc[:,iD]
    for i in cosName:
        temp1 = data_g.loc[:,i]
        tempData = pd.concat([tempData,temp1],axis=1)
    
    temp2 = tempData.values.T       
    cosData = cosine_similarity(temp2)
    
    ###########
    temp3 = [iD]
    temp4 = list(cosName)
    temp3.extend(temp4)
    
    df = pd.DataFrame(cosData,columns=temp3,index=temp3)
 
    df.to_csv(cd+'/med_knowledge/plot_networkx/'+iD+'/G'+str(g)+'_'+iD+'_top'+str(k)+'_cosCorrMat.csv',float_format='%.4f')
    
    return df


'===main==='
g = 4
iD = 'MED1027'
k = 10 # top k most relevent feature of iD
cosM = compute_cosCorrM(g,iD,k)

#################################################################################
##=========== plot networkx ===========
import networkx as nx
import matplotlib.pyplot as plt

####################################################################

def draw_networkx(cosM):
    G = nx.Graph()
    G.clear()

    iDname = list(cosM.columns)
    for i in iDname:
        for j in iDname:
            if i == j:
                continue
            else:
                weight = cosM.loc[i,j]
                G.add_weighted_edges_from([(i,j,weight)])
            
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

    pos = nx.circular_layout(G)# spring_layout (by default), random_layout, circular_layout, shell_layout
 
    nx.draw(G,pos,
            with_labels=True, font_size=10, 
            node_size=800, node_color=range(k+1),#node_count
            cmap=plt.cm.Set3,#spring
            #cmap=plt.cm.Blues,
            edgelist=edges, edge_color=weights, width=5.0, edge_cmap=plt.cm.Greys,
            )
    nx.draw_networkx_nodes(G,pos,nodelist=[iD],node_size=1200,node_color='r')
    
    #plt.savefig(cd+'/med_knowledge/plot_networkx/'+iD+'/G'+str(g)+'_'+iD+'.eps')
    plt.savefig(cd+'/med_knowledge/plot_networkx/'+iD+'/G'+str(g)+'_'+iD+'.png',dpi=400,bbox_inches='tight')
    


'===main===='
draw_networkx(cosM)












