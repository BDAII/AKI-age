# -*- coding: utf-8 -*-


import pandas as pd

cd = 'F:/AKI_age/AKI_judgment/'

g1_df = pd.read_excel(cd+'/Judgment_result/excel/'+"top150_group1.xlsx")
g2_df = pd.read_excel(cd+'/Judgment_result/excel/'+"top150_group2.xlsx")
g3_df = pd.read_excel(cd+'/Judgment_result/excel/'+"top150_group3.xlsx")
g4_df = pd.read_excel(cd+'/Judgment_result/excel/'+"top150_group4.xlsx")

##############################################################
#'=========input med judgment'
temp_df = pd.read_excel(cd+'/HI_result/doctor_xxx/'+'med.xls')

for i in temp_df.index:
    if(temp_df.loc[i , 'N1/N0'] == 'N1' or temp_df.loc[i , 'N1/N0'] == 'N0'):
        temp_df.loc[i , 'Judgment'] = temp_df.loc[i , 'N1/N0']
        


def fill_med_Judgment(df , temp_df):
    for i in range(len(df)):
        if df.loc[i , 'ID'].startswith("MED"):
            for j in range(len(temp_df)):
                if (df.loc[i, 'ID'] == temp_df.loc[j, 'medID']):
                    df.loc[i, 'Judgment'] = temp_df.loc[j, 'Judgment']
    return df



g1_df = fill_med_Judgment(g1_df , temp_df)
g2_df = fill_med_Judgment(g2_df , temp_df)
g3_df = fill_med_Judgment(g3_df , temp_df)
g4_df = fill_med_Judgment(g4_df , temp_df)

#####################################################################
#'=========input other binary variable and length_of_stay judgment'

other_temp_df = pd.read_excel(cd+'/HI_result/doctor_xxx/'+'binaryFeature.xlsx')


for i in temp_df.index:
    if(temp_df.loc[i , 'N1/N0'] == 'N1' or temp_df.loc[i , 'N1/N0'] == 'N0'):
        temp_df.loc[i , 'Judgment'] = temp_df.loc[i , 'N1/N0']
        

def fill_binary_Judgment(df , temp_df):
    for i in range(len(df)):
        if not (df.loc[i , 'ID'].startswith("MED")):
            for j in range(len(temp_df)):
                if (df.loc[i, 'ID'] == temp_df.loc[j, 'ID']):
                    df.loc[i, 'Judgment'] = temp_df.loc[j, 'Judgment']
    return df



g1_df = fill_binary_Judgment(g1_df , other_temp_df)
g2_df = fill_binary_Judgment(g2_df , other_temp_df)
g3_df = fill_binary_Judgment(g3_df , other_temp_df)
g4_df = fill_binary_Judgment(g4_df , other_temp_df)



'======== save excel'
g1_df.to_excel(cd+'/Judgment_result/doctor_xxx/'+'top150_group1.xlsx', header=True , index = False)
g2_df.to_excel(cd+'/Judgment_result/doctor_xxx/'+'top150_group2.xlsx', header=True , index = False)
g3_df.to_excel(cd+'/Judgment_result/doctor_xxx/'+'top150_group3.xlsx', header=True , index = False)
g4_df.to_excel(cd+'/Judgment_result/doctor_xxx/'+'top150_group4.xlsx', header=True , index = False)





