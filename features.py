import pandas as pd
import numpy as np
import datetime


def simple_features(df):

    # Make a copy of the DataFrame to avoid the SettingWithCopyWarning
    df = df.copy()
    
    df = df[(df['bedrooms'] != 0) & (df['bathrooms'] != 0)]

    # Assuming df is your original DataFrame
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    current_year = datetime.datetime.now().year
    df['property_age'] = current_year - df['year_built']

    df = pd.get_dummies(df, columns=['fips'], prefix='county')

    df['size_per_bedroom'] = df['area'] / df['bedrooms']

    df['bathroom_to_bedroom_ratio'] = df['bathrooms'] / df['bedrooms']

    return df


def age_cat(df):
    # property_age_group
    property_age_mean = df['property_age'].mean()
    property_age_std = df['property_age'].std()

    # Add corrected age range information for each category
    age_ranges = {
        'very_new': f'<= {int(property_age_mean - 2 * property_age_std)} years',
        'new': f'{int(property_age_mean - 2 * property_age_std) + 1} - {int(property_age_mean - property_age_std)} years',
        'mid-aged': f'{int(property_age_mean - property_age_std) + 1} - {int(property_age_mean + property_age_std)} years',
        'old': f'{int(property_age_mean + property_age_std) + 1} - {int(property_age_mean + 2 * property_age_std)} years',
        'very_old': f'>{int(property_age_mean + 2 * property_age_std)} years'
    }

    bins = [-float('inf'), property_age_mean - 2 * property_age_std, property_age_mean - property_age_std, property_age_mean + property_age_std, property_age_mean + 2 * property_age_std, float('inf')]

    labels = ['very_new', 'new', 'mid-aged', 'old', 'very_old']

    df['property_age_group'] = pd.cut(df['property_age'], bins=bins, labels=labels)
    
    return df


def size_cat(df):
    # property_size_category
    area_mean = df['area'].mean()
    area_std = df['area'].std()

    small_range = (0, area_mean - area_std)
    medium_range = (area_mean - area_std, area_mean + area_std)
    large_range = (area_mean + area_std, np.inf)  # No upper limit for 'large'

    df['property_size_category'] = pd.cut(df['area'], bins=[-np.inf, small_range[1], large_range[0], np.inf], labels=['small', 'medium', 'large'])

    return df

def make_features(df):
    df = simple_features(df)
    df = age_cat(df)
    df = size_cat(df)
    return df