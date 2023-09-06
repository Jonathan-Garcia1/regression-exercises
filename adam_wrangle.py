import os
import pandas as pd

from env import get_connection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_grades():
    
    filename = 'grades.csv'
    
    if os.path.isfile(filename):
        
        return pd.read_csv(filename)
    
    else:
        
        query = '''
                SELECT *
                FROM student_grades
                '''
        
        url = get_connection('school_sample')
        
        df = pd.read_sql(query, url)
        
        df.to_csv(filename, index=False)
        
        return df
    
    
def clean_grades():
    
    df = get_grades()
    
    df = df.dropna()
    
    df = df[df.exam3 != ' ']
    
    df.exam3 = df.exam3.astype('int')
    
    return df    


def train_val_test(df, seed = 42):
    
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed)
    
    return train, val, test


def xy_split(df):
    
    return df.drop(columns=['price']), df.price


def scale_data(train, val, test, to_scale):
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(val[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled