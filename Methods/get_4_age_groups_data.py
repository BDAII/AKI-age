# -*- coding: utf-8 -*-


import pickle
import pandas as pd


def get_4_age_groups_data(rawData): 
    '====step 1: since only consider AKI/NONAKI, so AKI 123 stages turn to aki=1====='
    rawData.label[rawData.label>=2]=1 # AKI stage 1,2,3 to 1


    '====step 2: split data for four age groups=====================================' 
    "18-35, 36-55,56-65, >65. i.e., 1+2, 3+4, 5, 6"
    data1 = rawData[rawData.age<3]
    data1.loc[:,'age']=1
    data2 = rawData[(rawData.age==3) | (rawData.age==4)]
    data2.loc[:,'age']=2
    data3 = rawData[rawData.age==5]
    data3.loc[:,'age']=3
    data4 = rawData[rawData.age==6]
    data4.loc[:,'age']=4


    '====step 3: delete columns with all values = 0  of univariate feature selection==============' 
    data1=data1.loc[:,~((data1==0).all())]; del data1['age']
    data2=data2.loc[:,~((data2==0).all())]; del data2['age']
    data3=data3.loc[:,~((data3==0).all())]; del data3['age']
    data4=data4.loc[:,~((data4==0).all())]; del data4['age']

    Data = [data1, data2, data3, data4]
    
    return Data


if __name__== '__main__':
    cd = '.../AKI_age/'
    findPath = cd + '/RawData/'
    savePath = cd + '/methods/'
    rawData = pd.read_csv(findPath + 'DataExtraction_2.csv')
    del rawData['Unnamed: 0']
    Data = get_4_age_groups_data(rawData)
     
    file=open(savePath+'Data.pkl','wb')
    pickle.dump(Data,file)
    file.close()

   
