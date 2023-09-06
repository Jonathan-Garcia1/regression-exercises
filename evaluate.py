# Imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from math import sqrt

from sklearn.linear_model import LinearRegression


def create_model(train, val):
    
    # split into subsets for train and val
    X_train = train[['area']]
    y_train = train.tax_value

    X_val = val[['area']]
    y_val = val.tax_value
    
    return X_train, y_train, X_val, y_val

def dataframe_model(X_train, y_train):
    # Since we have no way of knowing if the median or the mean performs better, we'll use the median.
    #y_train.median()
    
    preds = pd.DataFrame({'y_actual': y_train,
                        'y_baseline': y_train.median()})
    
    #baseline calculations residuals, residuals squared, baselie minus(-) actual
    preds['y_baseline_resids'] = preds['y_baseline'] - preds['y_actual']
    
    preds['y_baseline_resids_sq'] = preds['y_baseline_resids'] ** 2
    
    preds['y_baseline_minus_mean'] = preds['y_baseline'] - preds['y_actual'].mean()
    
    # assign the model and fit model
    lr = LinearRegression().fit(X_train, y_train)

    #save predictions
    preds['y_hat'] = lr.predict(X_train)
    
    # calculate the redisuals
    preds['y_hat_resids'] = preds['y_hat'] - preds['y_actual']
    
    # Now we square said residuals
    preds['y_hat_resids_sq'] = preds['y_hat_resids'] ** 2   
    
    # first lets calculate the prediction minus(-) actual
    preds['y_hat_minus_mean'] = preds['y_hat'] - preds['y_actual'].mean()
    
    return preds, lr

def plot_residuals(df):
    # creates a residual plot
    sns.scatterplot(data = df, x = 'y_actual', y = 'y_hat_resids')
    plt.show()

def regression_errors(df):
    # sum of squared errors (SSE)
    # add all the squared residuals to get the SSE
    sse_model = df['y_hat_resids_sq'].sum()
    
    # explained sum of squares (ESS)
    #add the new column then square it to get ESS
    ess_model = sum(df['y_hat_minus_mean'] ** 2)
    
    # total sum of squares (TSS)
    tss_model = sse_model + ess_model
    # mean squared error (MSE)
    mse_model = sse_model/len(df)
    # root mean squared error (RMSE)
    rmse_model = sqrt(mse_model)
    
    print(f'SSE:{sse_model:.3f}, ESS:{ess_model:.3f}, TSS:{tss_model:.3f}, RMSE:{rmse_model:.3f}')

    
    return sse_model, ess_model, tss_model, rmse_model

def baseline_mean_errors(df):
    #computes the SSE, MSE, and RMSE for the baseline model
    
    # sum of squared errors (SSE)
    # add all the squared residuals to get the SSE
    sse_baseline = df['y_baseline_resids_sq'].sum()
    
    # explained sum of squares (ESS)
    #add the column then square it to get ESS
    ess_baseline = sum(df['y_baseline_minus_mean'] ** 2)
    
    # total sum of squares (TSS)
    tss_baseline = sse_baseline + ess_baseline
    # mean squared error (MSE)
    mse_baseline = sse_baseline/len(df)
    # root mean squared error (RMSE)
    rmse_baseline = sqrt(mse_baseline)
    
    print(f'SSE:{sse_baseline:.3f}, ESS:{ess_baseline:.3f}, TSS:{tss_baseline:.3f}, RMSE:{rmse_baseline:.3f}')

    
    return sse_baseline, ess_baseline, tss_baseline, rmse_baseline


def better_than_baseline(df):
    #returns true if your model performs better than the baseline, otherwise false
    
    sse_baseline = df['y_baseline_resids_sq'].sum()
    sse_model = df['y_hat_resids_sq'].sum()
    
    if sse_baseline > sse_model:
        print(f'Good, SSE of our finely tuned model: {round(sse_model)} beat the baseline {round(sse_baseline)}')
    else:
        print(f'Bad, SSE of our baseline: {round(sse_baseline)} beat the tuned model: {round(sse_model)}')
    return