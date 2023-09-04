import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from env import get_connection
import os

def acquire_zillow():
    # create helper function to get the necessary connection url.
    def get_db_connection(database):
        return get_connection(database)

    # connect to sql zillow database
    url = "zillow"

    # use this query to get data    
    sql_query = '''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
                FROM properties_2017 
                WHERE propertylandusetypeid = 261
                '''

    # assign data to data frame
    df = pd.read_sql(sql_query, get_connection(url))

    return df

def wrangle_zillow(df):
    # Drop nulls
    df = df.dropna()
    
    # Rename the columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area', 
                              'taxvaluedollarcnt':'tax_value', 'yearbuilt':'year_built'})

    # Convert data types
    df.bedrooms = df.bedrooms.astype('int')
    df.area = df.area.astype('int')
    df.tax_value = df.tax_value.astype('int')
    df.year_built = df.year_built.astype('int')
    df.fips = df.fips.astype('int')

    # Save to csv
    df.to_csv('zillow_data.csv',index=False)

    return df 

def scale_train_val_test(train, val, test):

    mms = MinMaxScaler()

    mms.fit(train[['year_built']])

    train['year_built'] = mms.transform(train[['year_built']])
    val['year_built'] = mms.transform(val[['year_built']])
    test['year_built'] = mms.transform(test[['year_built']])
    
    return train, val, test

