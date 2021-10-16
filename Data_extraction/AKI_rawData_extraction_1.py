# -*- coding: utf-8 -*-

"""
The data extracted from the database was saved as TXT text. 
This code file is used to extract TXT text information 
and store it as a nested list to generate pickle files for subsequent reprocessing.
"""

import pickle

def get_data(filepath):
           
    f=open(filepath + 'RawData.txt','r')
    lines=f.readlines() 
    f.close()
    
    return lines


def static_information(data):
    ''' 
    Information storage format:'value0_value1_...' 
    Static information has only the first level of separation (namely '_')
    After extraction, it is saved as a 1-dimensional list, [value0,value1,...]
    data type: int                   
    '''
    
    temp=data.split('_')
    result=list(map(int,temp))
    
    return result


def temporal_information(data):
    '''
    Information storage format：'{[(),(),...];[];...}_{}_...'
    the first level of separation: '_',
    First-level separator: '_'
    Second-level separator: ';' 
    Third-level separator: ','
    After extraction, it is saved as a 3-dimensional list,[[[value,value,...],[],...],[],...]
    '''
    
    first_split=data.split('_')
    second_split=[first_split[i].split(';') for i in range(len(first_split))]
    third_split=[[second_split[i][j].split(',') for j in range(len(second_split[i]))] for i in range(len(second_split))]
    
    result=[[[int(third_split[i][j][k]) for k in range(len(third_split[i][j]))] for j in range(len(third_split[i]))] for i in range(len(third_split))]
    
    return result

      
def label_information(data):
    '''
    Information storage format：'label,date\n'
    save format：[label,date],
    data type: int
    '''
    
    result=data.split(',')
    result[0]=int(result[0])
    result[1]=int(result[1].strip('\n'))
    
    return result


def combine_imed_omed(imed_data,omed_data):
    ##Combining in-hospital and out-of-hospital medication information
    
    temp=omed_data[:]
    
    for j in range(len(imed_data)):
        flag=bool(1)
        for k in range(len(omed_data)):
            if imed_data[j][0]==omed_data[k][0]: 
                flag=bool(0)
                try:
                    temp[k][1]=list(set(imed_data[j][1])|set(omed_data[k][1]))
                    temp[k][1].sort()
                except IndexError: continue
                else: break
        if flag: temp.append(imed_data[j])
    
    return temp
                       
def main_function(filepath):
    lines = get_data(filepath)
    
    data_split_category=[]
    temp=[data_split_category.append(lines[i].split('|')) for i in range(len(lines))] 
    
    demo=[] # demographics information
    vitals=[] # vital signs
    lab=[] #lab tests
    drg=[] #admission diagnoses 
    imed=[] # Inpatient medication referred to drugs dispensed during hospitalization
    omed=[] # outpatient medication included outpatient prescriptions and drugs taken at home
    med=[] # medications 
    ccs=[] # medical history
    label=[] # AKI or non-AKI

    for i in range(len(data_split_category)):
    
        demo.append(static_information(data_split_category[i][0]))
        vitals.append(temporal_information(data_split_category[i][1]))
        lab.append(temporal_information(data_split_category[i][2]))    
        drg.append(static_information(data_split_category[i][3]))
 
        imed.append(temporal_information(data_split_category[i][4]))
        omed.append(temporal_information(data_split_category[i][5]))
        med.append(combine_imed_omed(imed[i],omed[i])) for i in range(len(data_split_category))
        
        ccs.append(temporal_information(data_split_category[i][6]))
        
        label.append(label_information(data_split_category[i][7]))

    Data=[demo,vitals,lab,drg,med,ccs,label]
    
    return Data

def save_data(filePath, fileName, Data):    
    ## save data into a pickle file
    file=open(filePath + fileName +'.pkl','wb')
    pickle.dump(Data,file)
    file.close()
        
if __name__== '__main__':
    filePath = '.../AKI_age/RawData/'
    fileName = 'DataExtraction_1'
    Data = main_function(filePath)
    save_data(filepath, Data)

