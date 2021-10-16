# -*- coding: utf-8 -*-


import pickle
import numpy as np
import pandas as pd



def static_demo(demo):
    '''
    static demo information: age, race, gender
    the number of features: 3
    '''

    demo0=np.array(demo) 

    data ={'age': list(demo0[:,0]),
           'race': list(demo0[:,1]),
           #'gender{1,2}变成{0，1},namely {female, male}'
           'gender': list(demo0[:,2]-1)
          }
          
    demo1 = pd.DataFrame(data)

    # race, columns=['White','African American','Asian','Other race']
    dummies = pd.get_dummies(demo1['race'], prefix='race')

    Demo = demo1[['age','gender']].join(dummies)

    return Demo


def static_drg(drg):
    '''
    static drg information
    the number of features: 315
    '''
    
    drg0=np.zeros([len(drg), 315],dtype=np.int)
 
    for i in range(len(drg)):
        drg0[i,drg[i]]=1 
        
    Drg = pd.DataFrame(drg0).add_prefix('DRG')

    return Drg



'==========latest'
def get_vitals_lab_latest(input_data,t_time):
    value=np.zeros([len(input_data),len(input_data[0])],dtype=np.int) # vitals is 5
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            try:  
                temp=input_data[i][j] # for ith sample and jth feature
                temp1=np.asarray(temp) #[[value,day],[value,day]]
                temp2=temp1[:,-1] # [day1,day2,day3,..]
                temp3=[x for x in temp2 if x<=t_time[i]]
                temp4=np.max(temp3)
                temp5=list(temp2).index(temp4)
                value[i,j]=temp[temp5][0]
            except: continue   
            
    return value



def temporal_vitals(vitals,t_time):
    '''
    temporal vital signs information
    the number of features: 5
    '''
    
    Vitals_latest0 = get_vitals_lab_latest(vitals,t_time)  
    Vitals_latest = pd.DataFrame(Vitals_latest0, 
                             columns=['BMI','DBP','SBP','Pulse','Temp'])      
    
    ######################
    # 1) In order to enhance the interpretability of the subsequent analysis, 
    # in this study, we choose to convert the multi-valued variables (vitals and labs) into multiple binary variables.
    # 2) Missing values may also represent unexamined information, so we chose to use the missing value as a separate variable.
    # 3) In the code, we choose to directly extract and transform here, 
    # or you can extract data first and transform it in the subsequent data preprocessing stage.
    #######################

        
    ########## BMI={0,1,2,3,4}, 
    temp = Vitals_latest['BMI']
    temp1 = pd.concat([temp,temp,temp,temp,temp],axis=1)#,ignore_index=True
    temp1.columns = ['BMI_0','BMI_1','BMI_2','BMI_3','BMI_4']

    temp2 = temp1[(temp1['BMI_0']==0)].index.tolist()
    temp3 = temp1[(temp1['BMI_0']!=0)].index.tolist()
    temp1.loc[temp3,'BMI_0']=0               
    temp1.loc[temp2,'BMI_0']=1 

    temp2 = temp1[(temp1['BMI_1']==1)].index.tolist()
    temp3 = temp1[(temp1['BMI_1']!=1)].index.tolist()
    temp1.loc[temp3,'BMI_1']=0               
    temp1.loc[temp2,'BMI_1']=1 

    temp2 = temp1[(temp1['BMI_2']==2)].index.tolist()
    temp3 = temp1[(temp1['BMI_2']!=2)].index.tolist()
    temp1.loc[temp3,'BMI_2']=0               
    temp1.loc[temp2,'BMI_2']=1 

    temp2 = temp1[(temp1['BMI_3']==3)].index.tolist()
    temp3 = temp1[(temp1['BMI_3']!=3)].index.tolist()
    temp1.loc[temp3,'BMI_3']=0               
    temp1.loc[temp2,'BMI_3']=1 

    temp2 = temp1[(temp1['BMI_4']==4)].index.tolist()
    temp3 = temp1[(temp1['BMI_4']!=4)].index.tolist()
    temp1.loc[temp3,'BMI_4']=0               
    temp1.loc[temp2,'BMI_4']=1 

    df_BMI = temp1[:]

    ############ DBP={0,1,2,3,4}
    temp = Vitals_latest['DBP']
    temp1 = pd.concat([temp,temp,temp,temp,temp],axis=1)#,ignore_index=True
    temp1.columns = ['DBP_0','DBP_1','DBP_2','DBP_3','DBP_4']

    temp2 = temp1[(temp1['DBP_0']==0)].index.tolist()
    temp3 = temp1[(temp1['DBP_0']!=0)].index.tolist()
    temp1.loc[temp3,'DBP_0']=0               
    temp1.loc[temp2,'DBP_0']=1 

    temp2 = temp1[(temp1['DBP_1']==1)].index.tolist()
    temp3 = temp1[(temp1['DBP_1']!=1)].index.tolist()
    temp1.loc[temp3,'DBP_1']=0               
    temp1.loc[temp2,'DBP_1']=1 

    temp2 = temp1[(temp1['DBP_2']==2)].index.tolist()
    temp3 = temp1[(temp1['DBP_2']!=2)].index.tolist()
    temp1.loc[temp3,'DBP_2']=0               
    temp1.loc[temp2,'DBP_2']=1 

    temp2 = temp1[(temp1['DBP_3']==3)].index.tolist()
    temp3 = temp1[(temp1['DBP_3']!=3)].index.tolist()
    temp1.loc[temp3,'DBP_3']=0               
    temp1.loc[temp2,'DBP_3']=1 

    temp2 = temp1[(temp1['DBP_4']==4)].index.tolist()
    temp3 = temp1[(temp1['DBP_4']!=4)].index.tolist()
    temp1.loc[temp3,'DBP_4']=0               
    temp1.loc[temp2,'DBP_4']=1 

    df_DBP = temp1[:]

    ########### SBP={0,1,2,3,4}
    temp = Vitals_latest['SBP']
    temp1 = pd.concat([temp,temp,temp,temp,temp],axis=1)#,ignore_index=True
    temp1.columns = ['SBP_0','SBP_1','SBP_2','SBP_3','SBP_4']

    temp2 = temp1[(temp1['SBP_0']==0)].index.tolist()
    temp3 = temp1[(temp1['SBP_0']!=0)].index.tolist()
    temp1.loc[temp3,'SBP_0']=0               
    temp1.loc[temp2,'SBP_0']=1 

    temp2 = temp1[(temp1['SBP_1']==1)].index.tolist()
    temp3 = temp1[(temp1['SBP_1']!=1)].index.tolist()
    temp1.loc[temp3,'SBP_1']=0               
    temp1.loc[temp2,'SBP_1']=1 

    temp2 = temp1[(temp1['SBP_2']==2)].index.tolist()
    temp3 = temp1[(temp1['SBP_2']!=2)].index.tolist()
    temp1.loc[temp3,'SBP_2']=0               
    temp1.loc[temp2,'SBP_2']=1 

    temp2 = temp1[(temp1['SBP_3']==3)].index.tolist()
    temp3 = temp1[(temp1['SBP_3']!=3)].index.tolist()
    temp1.loc[temp3,'SBP_3']=0               
    temp1.loc[temp2,'SBP_3']=1 

    temp2 = temp1[(temp1['SBP_4']==4)].index.tolist()
    temp3 = temp1[(temp1['SBP_4']!=4)].index.tolist()
    temp1.loc[temp3,'SBP_4']=0               
    temp1.loc[temp2,'SBP_4']=1 

    df_SBP = temp1[:]

    ############### Pulse={0,1,2,3,4,5}
    temp = Vitals_latest['Pulse']
    temp1 = pd.concat([temp,temp,temp,temp,temp,temp],axis=1)#,ignore_index=True
    temp1.columns = ['Pulse_0','Pulse_1','Pulse_2','Pulse_3','Pulse_4','Pulse_5']

    temp2 = temp1[(temp1['Pulse_0']==0)].index.tolist()
    temp3 = temp1[(temp1['Pulse_0']!=0)].index.tolist()
    temp1.loc[temp3,'Pulse_0']=0               
    temp1.loc[temp2,'Pulse_0']=1 
    
    temp2 = temp1[(temp1['Pulse_1']==1)].index.tolist() 
    temp3 = temp1[(temp1['Pulse_1']!=1)].index.tolist()
    temp1.loc[temp3,'Pulse_1']=0               
    temp1.loc[temp2,'Pulse_1']=1 

    temp2 = temp1[(temp1['Pulse_2']==2)].index.tolist()
    temp3 = temp1[(temp1['Pulse_2']!=2)].index.tolist()
    temp1.loc[temp3,'Pulse_2']=0               
    temp1.loc[temp2,'Pulse_2']=1 

    temp2 = temp1[(temp1['Pulse_3']==3)].index.tolist()
    temp3 = temp1[(temp1['Pulse_3']!=3)].index.tolist()
    temp1.loc[temp3,'Pulse_3']=0               
    temp1.loc[temp2,'Pulse_3']=1 

    temp2 = temp1[(temp1['Pulse_4']==4)].index.tolist()
    temp3 = temp1[(temp1['Pulse_4']!=4)].index.tolist()
    temp1.loc[temp3,'Pulse_4']=0               
    temp1.loc[temp2,'Pulse_4']=1 

    temp2 = temp1[(temp1['Pulse_5']==5)].index.tolist()
    temp3 = temp1[(temp1['Pulse_5']!=5)].index.tolist()
    temp1.loc[temp3,'Pulse_5']=0               
    temp1.loc[temp2,'Pulse_5']=1 

    df_Pulse = temp1[:]

    ############# 'Temperature={0,1,2,3,4,5}'
    ## We adjusted the order of the instructions in the table appropriately (see DataFeature_Readme.docx)    

    temp = Vitals_latest['Temp']
    temp1 = pd.concat([temp,temp,temp,temp,temp],axis=1)#,ignore_index=True
    temp1.columns = ['Temperature_0','Temperature_1','Temperature_2','Temperature_3','Temperature_4',,'Temperature_5']

    temp2 = temp1[(temp1['Temperature_0']==0)].index.tolist()
    temp3 = temp1[(temp1['Temperature_0']!=0)].index.tolist()
    temp1.loc[temp3,'Temperature_0']=0               
    temp1.loc[temp2,'Temperature_0']=1 

    temp2 = temp1[(temp1['Temperature_1']==1)].index.tolist()
    temp3 = temp1[(temp1['Temperature_1']!=1)].index.tolist()
    temp1.loc[temp3,'Temperature_1']=0               
    temp1.loc[temp2,'Temperature_1']=1 

    temp2 = temp1[(temp1['Temperature_2']==2)].index.tolist()
    temp3 = temp1[(temp1['Temperature_2']!=2)].index.tolist()
    temp1.loc[temp3,'Temperature_2']=0               
    temp1.loc[temp2,'Temperature_2']=1 

    temp2 = temp1[(temp1['Temperature_3']==3)].index.tolist()
    temp3 = temp1[(temp1['Temperature_3']!=3)].index.tolist()
    temp1.loc[temp3,'Temperature_3']=0               
    temp1.loc[temp2,'Temperature_3']=1 

    temp2 = temp1[(temp1['Temperature_4']==4)].index.tolist()
    temp3 = temp1[(temp1['Temperature_4']!=4)].index.tolist()
    temp1.loc[temp3,'Temperature_4']=0               
    temp1.loc[temp2,'Temperature_4']=1 

    temp2 = temp1[(temp1['Temperature_5']==5)].index.tolist()
    temp3 = temp1[(temp1['Temperature_5']!=5)].index.tolist()
    temp1.loc[temp3,'Temperature_5']=0               
    temp1.loc[temp2,'Temperature_5']=1 

    df_Temperature = temp1[:]

    ###' vital signs'
    Vitals = pd.concat([df_BMI,df_DBP,df_SBP,df_Pulse,df_Temperature], axis=1)

    return Vitals


def temporal_lab(lab,t_time):
    '''
    temporal lab test information
    the number of features: 14
    '''
   
    Lab_latest0 = get_vitals_lab_latest(lab,t_time)
    Lab_latest = pd.DataFrame(Lab_latest0).add_prefix('Lab')

    def lab_0123(temp):
        #'Lab={0,1,2,3}'
        temp1 = pd.concat([temp,temp,temp,temp],axis=1)
        temp2 = temp1.columns.tolist()
        name = temp2[0]
        temp1.columns = [name+'_0',name+'_1',name+'_2',name+'_3']
    
        temp2 = temp1[(temp1[name+'_0']==0)].index.tolist()
        temp3 = temp1[(temp1[name+'_0']!=0)].index.tolist()
        temp1.loc[temp3,name+'_0']=0               
        temp1.loc[temp2,name+'_0']=1 
    
        temp2 = temp1[(temp1[name+'_1']==1)].index.tolist()
        temp3 = temp1[(temp1[name+'_1']!=1)].index.tolist()
        temp1.loc[temp3,name+'_1']=0               
        temp1.loc[temp2,name+'_1']=1 

        temp2 = temp1[(temp1[name+'_2']==2)].index.tolist()
        temp3 = temp1[(temp1[name+'_2']!=2)].index.tolist()
        temp1.loc[temp3,name+'_2']=0               
        temp1.loc[temp2,name+'_2']=1 

        temp2 = temp1[(temp1[name+'_3']==3)].index.tolist()
        temp3 = temp1[(temp1[name+'_3']!=3)].index.tolist()
        temp1.loc[temp3,name+'_3']=0               
        temp1.loc[temp2,name+'_3']=1 
    
        return temp1


    Lab=pd.DataFrame(columns=["nan"])
    for i in range(len(Lab_latest.columns)):
        temp0 = Lab_latest.iloc[:,i]
        df_lab = lab_0123(temp0)
        Lab = pd.concat([Lab,df_lab],axis=1)

    del Lab['nan']
    
    return Lab
    

def temporal_css(ccs, t_time):
    '''
    temporal ccs information
    the number of features: 280
    get data: 1 if present, 0 otherwise
    '''


    def get_ccs(input_data,t_time):
        value=np.zeros([len(input_data),280],dtype=np.int)
        for i in range(len(input_data)):
            for j in range(len(input_data[i])):
                flag=bool(0)
                for z in input_data[i][j][-1]:
                    if z <= t_time[i]:
                        flag=bool(1)
                        break
                if flag: value[i,input_data[i][j][0]]=1
        return value
                
    Ccs0=get_ccs(ccs,t_time)
    Ccs = pd.DataFrame(Ccs0).add_prefix('CCS')

    return Ccs



def temporal_med(med, t_time):
    '''
    temporal med information
    the number of features: 1271
    Taking into account drug metabolism,
    we used the cumulative-exposure-days of medications during the 7-day window before the reference point as predictors
    '''

    def get_med_week(input_data,t_time, med_window):
        value_week = np.zeros([len(input_data),1271],dtype=np.int)
        for i in range(len(input_data)):
            for j in range(len(input_data[i])):
                temp = input_data[i][j][-1]
                temp_week = [x for x in temp if x in range(t_time[i]-med_window+1, t_time[i]+1)]
                days_week = len(temp_week)
                value_week[i,input_data[i][j][0]]=days_week
        return value_week

               
    Med0 = get_med_week(med,t_time,med_window)
    Med = pd.DataFrame(Med0).add_prefix('MED')  #.add_suffix('_week')

    return Med



#############################################################
####################### other history code ##################
#############################################################

def get_vitals_lab_max_min(input_data,t_time):
    value_max=np.zeros([len(input_data),len(input_data[0])],dtype=np.int) 
    value_min=np.zeros([len(input_data),len(input_data[0])],dtype=np.int) 
    
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            try:  
                temp=input_data[i][j] # for ith sample and jth feature
                temp1 = [x for x in temp if x[-1]<=t_time[i]]
                temp2 = np.array(temp1)
                value_max[i,j] = np.max(temp2[:,0])
                value_min[i,j] = np.min(temp2[:,0])
            
            except: continue    
            
    return [value_max, value_min]
        



def get_med_history(input_data,t_time, med_window):
    value_history = np.zeros([len(input_data),1271],dtype=np.int)
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            temp = input_data[i][j][-1]
            temp_history = [x for x in temp if x <= t_time[i]]
            days_history = len(temp_history)
            value_history[i,input_data[i][j][0]]=days_history

    return value_history
 

 
def get_med_week_history(input_data,t_time, med_window):
    value_week = np.zeros([len(input_data),1271],dtype=np.int)
    value_history = np.zeros([len(input_data),1271],dtype=np.int)
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            temp = input_data[i][j][-1]
            temp_week = [x for x in temp if x in range(t_time[i]-med_window+1, t_time[i]+1)]
            temp_history = [x for x in temp if x <= t_time[i]]
            days_week = len(temp_week)
            days_history = len(temp_history)
            value_week[i,input_data[i][j][0]]=days_week
            value_history[i,input_data[i][j][0]]=days_history

    return [value_week, value_history]

  
#############################################################
#############################################################


def main_function(pre_day, med_window):
    filePath = '.../AKI_age/RawData/'
    fileName = 'DataExtraction_1.pkl'
    f = open(filePath + fileName, 'rb')
    data = pickle.load(f)
    
    # feature category
    demo, vitals, lab, drg, med, ccs, label = data
    
    # get the corresponding time
    a=np.array(label)
    y_value=a[:,0] # the first aki event 
    t_time=a[:,1] - pre_day
    
    #######################
    #Because some features have no value, we can’t mine the scope of the data itself, 
    #we choose to define the zero matrix through the feature description given during data extraction.
    #######################
        
    Demo = static_demo(demo)
    Vitals = temporal_vitals(vitals, t_time)
    Lab = temporal_lab(lab, t_time)
    Drg = static_drg(drg)
    Ccs = temporal_css(ccs, t_time)
    Med = temporal_med(med, t_time)
    
    # length of stay
    Length_stay = pd.DataFrame(list(t_time[:]), columns=['days'])   
    # y label
    yValue = pd.DataFrame(list(y_value),columns=['label'])       
    
    final_list = [Demo, Vitals, Lab, Drg, Ccs, Med, Length_stay, yValue]
    final_Data = pd.concat(final_list, axis=1)
    
    return final_Data
    

def save_data(filePath, fileName, Data, type):
    if type == 'csv':
        Data.to_csv(filePath + fileName +'.csv')
    elif type == 'pkl':
        f =  open(filePath + fileName + '.pkl', 'wb')
        pickle.dump(Data, f)
        f.close()
    else:
        print('Error input: type (csv or pkl)')
    

    
if __name__== '__main__':
    pre_day = 0
    med_window = 7
    Data = main_function(pre_day, med_window)
    filePath = '.../AKI_age/RawData/'
    fileName = 'DataExtraction_2'
    save_data(filePath, fileName, Data, 'csv')
    
  












