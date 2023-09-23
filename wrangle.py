import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

def get_zillow_data():
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        # Read fresh data from db into a dataframe
        df = acquire_zillow()
        # Cache data
        df.to_csv('zillow.csv')
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

    df = df.drop(['taxamount'], axis=1)
    # Save to csv
    df.to_csv('zillow_data.csv',index=False)

    return df

def drop_zeros(df):
    df = df[(df['bedrooms'] != 0) & (df['bathrooms'] != 0)]
    return df

def zillow_pipeline():
    df = get_zillow_data()
    df = wrangle_zillow(df)
    df = drop_zeros(df)
    return df


def split_train_val_test(df):
    #split data
    seed = 42

    train, val_test = train_test_split(df, train_size=0.7,
                                    random_state=seed)

    val, test = train_test_split(val_test, train_size=0.5,
                                random_state=seed)
    return train, val, test


def scale_train_val_test(train, val, test):

    mms = MinMaxScaler()

    # Fit the scaler on the training data for all columns you want to scale
    columns_to_scale = ['year_built', 'area'] # 'taxamount',
    mms.fit(train[columns_to_scale])
    
    # Transform the specified columns for each dataset
    train[columns_to_scale] = mms.transform(train[columns_to_scale])
    val[columns_to_scale] = mms.transform(val[columns_to_scale])
    test[columns_to_scale] = mms.transform(test[columns_to_scale])
    
    return train, val, test


def scale_train_val_test2(train, val, test):

    mms = MinMaxScaler()

    # Fit the scaler on the training data for all columns you want to scale
    columns_to_scale = ['bedrooms', 'bathrooms', 'area', 'year_built',
       'total_rooms', 'property_age', 'county_6037', 'county_6059',
       'county_6111', 'size_per_bedroom', 'bathroom_to_bedroom_ratio',
       'property_size_category_small', 'property_size_category_medium',
       'property_size_category_large', 'property_age_group_very_new',
       'property_age_group_new', 'property_age_group_mid-aged',
       'property_age_group_old', 'property_age_group_very_old'] # 'taxamount',
    mms.fit(train[columns_to_scale])
    
    # Transform the specified columns for each dataset
    train[columns_to_scale] = mms.transform(train[columns_to_scale])
    val[columns_to_scale] = mms.transform(val[columns_to_scale])
    test[columns_to_scale] = mms.transform(test[columns_to_scale])
    
    return train, val, test

def xy_split(df):
    
    
    return df.drop(columns=['tax_value']), df.tax_value