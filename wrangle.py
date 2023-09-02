import pandas as pd

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