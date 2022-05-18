import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer

#acquire info:
import acquire as acq


#### DATA SPLITTING ####
def splitData(df,target,**kwargs):
    """
    Returns: 3 dataframes - train, test, validate
    Parameters:
      (R) df: dataframe to be split
      (R) target: Column name of the target variable - for stratifying
      (O -kw) val_ratio: Proportion of the whole dataset wanted for the validation subset (b/w 0 and 1)
      (O -kw) test_ratio: Proportion of the whole dataset wanted for the test subset (b/w 0 and 1)

    """
    ##ADD kwargs later that would allow you to specify a validation and test ratio
    #test and validation percentages of WHOLE dataset -
    val_per = kwargs.get('val_ratio',.2)
    test_per = kwargs.get('test_ratio',.1)

    #Calculate percentage we need of test/train subset
    tt_per = test_per/(1-val_per)

    #returns train then test, so test_size is the second set it returns
    tt, validate = train_test_split(df, test_size=val_per,random_state=88,stratify=df[target])
    #now split tt in train and test want 70/10 so test_size = 1/8 or .125
    train, test = train_test_split(tt, test_size=tt_per, random_state=88,stratify=tt[target])
    
    return train, test, validate


#### TITANIC DATASET ####

##Imputing age function
def impute_titanic_age(train,test,validate):
    imputer = SimpleImputer(strategy='mean')
    #fit train data to imputer, then transform train
    train['age'] = imputer.fit_transform(train[['age']])
    #transform test and validate
    test['age'] = imputer.transform(test[['age']])
    validate['age'] = imputer.transform(validate[['age']])
    return train, test, validate

##Outer prep function
def prep_titanic(df,**kwargs):
    #Drop un-needed columns
    df.drop(inplace=True,columns=['pclass','passenger_id','embarked','deck'])
    #Drop minimal nulls from embark town
    df.dropna(subset=['embark_town'],inplace=True)
    #Encode embark_town, sex and class >> concat
    d_df = pd.get_dummies(df[['embark_town','sex','class']],drop_first=[True, True])
    df = pd.concat([df,d_df],axis=1)
    #rename class for ease of coding
    df.rename(columns={'class':'pclass'},inplace=True)

    #Now split the data:
    target='survived'
    train, test, validate = splitData(df,target,**kwargs)

    #Impute age and return datasets
    return impute_titanic_age(train, test, validate)



#### IRIS DATASET ####
def prep_iris(idf,**kwargs):
    #drop id columns:
    idf.drop(inplace=True,columns=['species_id','measurement_id'])
    #rename species column:
    idf.rename(columns={"species_name":"species"},inplace=True)
    #encode species:
    d_idf = pd.get_dummies(idf[['species']],drop_first=True)
    #concat to dataframe
    idf = pd.concat([idf,d_idf],axis=1)

    #Now split the data:
    target='species'
    train, test, validate = splitData(idf,target,**kwargs)

    return train, test, validate


#### TELCO DATASET ####
def prep_telco(df,**kwargs):
       
    #HANDLE total_charge row:
    #grab the indices with null values
    drp_ind = df[df.total_charges.str.strip() == ''].index
    #drop those indices
    df.drop(index=drp_ind,inplace=True)
    #Convert the column to float
    df.total_charges = df.total_charges.astype(float)

    #DROP other unnecessary columns
    drp_col = ['payment_type_id','internet_service_type_id','contract_type_id','customer_id','signup_date']
    df.drop(columns = drp_col,inplace=True)

    #MAP subset of variables that are yes/no
    #phone_service, paperless_billing, partner, dependents, churn
    df['has_phone'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['is_paperless'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['has_dependents'] = df.partner.map({'Yes': 1, 'No': 0})
    df['has_partner'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['has_churned'] = df.churn.map({'Yes': 1, 'No': 0})   

    #ENCODE the other categorical columns
    enc_col = ['gender', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies','payment_type','contract_type','internet_service_type']  
    d_df = pd.get_dummies(df[enc_col],drop_first=True)
    #concate to df
    df = pd.concat([df,d_df],axis=1)

    #Now split the data:
    target='churn'
    train, test, validate = splitData(df,target,**kwargs)

    return train, test, validate