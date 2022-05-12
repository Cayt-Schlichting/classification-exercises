import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer

#acquire info:
import acquire as acq

#### TITANIC DATASET ####
def prep_titanic(df):
    #Drop un-needed columns
    df.drop(inplace=True,columns=['pclass','passenger_id','embarked','deck'])
    #Drop minimal nulls from embark town
    df.dropna(subset=['embark_town'],inplace=True)
    #Encode embark_town, sex and class >> concat
    d_df = pd.get_dummies(df[['embark_town','sex','class']],drop_first=[True, True])
    df = pd.concat([df,d_df],axis=1)
    #rename class for ease of coding
    df.rename(columns={'class':'pclass'},inplace=True)
    return df



#### IRIS DATASET ####
def prep_iris(idf):
    #drop id columns:
    idf.drop(inplace=True,columns=['species_id','measurement_id'])
    #rename species column:
    idf.rename(columns={"species_name":"species"},inplace=True)
    #encode species:
    d_idf = pd.get_dummies(idf[['species']],drop_first=True)
    #concat to dataframe
    idf = pd.concat([idf,d_idf],axis=1)
    return idf


#### TELCO DATASET ####