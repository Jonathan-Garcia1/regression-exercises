from sklearn.metrics import mean_squared_error
from math import sqrt


def eval_model(y_actual, y_hat):
    
    """Calculate the RMSE.
    
       Pass in the actual values first and the predicted values second."""
    
    return sqrt(mean_squared_error(y_actual, y_hat))


def train_model(model, X_train, y_train, X_val, y_val):
    
    """Train the model. Pass in the following arguments:
       
       Model
       X_train
       y_train
       X_val
       y_val"""
    
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    
    val_rmse = eval_model(y_val, val_preds)
    
    print(f'The train RMSE is {train_rmse}.')
    print(f'The validate RMSE is {val_rmse}.')
    
    return model