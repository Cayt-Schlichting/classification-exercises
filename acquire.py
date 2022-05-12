import os
import pandas as pd
from env import get_db_url

#Populate dataframe with filenames, sql and databases for select datasets
ind = ['titanic','iris','telco']
ds = pd.DataFrame(index=ind,columns=['filename','db_name','sql'])
ds.loc['titanic'] = ['titanic.csv','titanic_db',"""SELECT * FROM passengers;"""]
ds.loc['iris'] = ['iris.csv','iris_db',"""SELECT * FROM measurements JOIN species USING(species_id);"""]
telco_sql = """
SELECT * FROM customers
JOIN internet_service_types USING(internet_service_type_id)
JOIN contract_types USING(contract_type_id)
JOIN payment_types USING(payment_type_id)
JOIN customer_signups USING(customer_id);
"""
ds.loc['telco'] = ['telco.csv','telco_churn',telco_sql]

def getNewData(db,ds=ds):
    """
    Returns Pandas dataframe of desired dataset.
    Required Parameter: dataset name
    Supported datasets: 
      titanic
      iris
      telco
    
    Retrieves dataset from Codeup DB and stores a local csv file
    """
    db_name = ds.loc[db,'db_name']
    sql = ds.loc[db,'sql']
    filename = ds.loc[db,'filename']
    
    df = pd.read_sql(sql,get_db_url(db_name))
    #write to disk - writes index as col 0:
    df.to_csv(filename)
    return 


def getData(db,ds=ds):
    """
    Returns: Pandas dataframe of desired dataset.
    Required Parameter: dataset name
    Supported datasets: 
      titanic
      iris
      telco
    
    Checks working directory for csv file of data.
    If no CSV, retrieves dataset from Codeup DB and stores a local csv file
    """
    filename = ds.loc[db,'filename']

    if os.path.isfile(filename): #check if file exists in WD
        #grab data, set first column as index
        return pd.read_csv(filename,index_col=[0])
    else: #Get data from SQL db
        df = getNewData(db)
    return df

