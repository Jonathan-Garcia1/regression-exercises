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


def estimate_t_time(model, X_train, y_train, X_val, y_val, subsample_fraction=0.1):
    
    def format_time(seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02d}:{int(minutes):02d}"
    
    # Convert X_train and y_train to pandas DataFrames or Series if they are not already
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
        
    if not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(X_val)
    if not isinstance(y_val, pd.Series):
        y_val = pd.Series(y_val)
    
    # Determine the size of the subset based on the subsample_fraction
    subset_size = int(len(X_train) * subsample_fraction)
    
    # Create a random subset of the data for training
    random_indices_train = np.random.choice(len(X_train), size=subset_size, replace=False)
    X_subset_train = X_train.iloc[random_indices_train]
    y_subset_train = y_train.iloc[random_indices_train]

    # Create a random subset of the validation data
    random_indices_val = np.random.choice(len(X_val), size=subset_size, replace=False)
    X_subset_val = X_val.iloc[random_indices_val]
    y_subset_val = y_val.iloc[random_indices_val]
    
    # Record the start time
    start_time = time.time()
    
    # Train the model on the subset of the training data
    train_model(model, X_subset_train, y_subset_train, X_subset_val, y_subset_val)
    
    # Calculate the elapsed time for training the subset
    elapsed_time_subset = time.time() - start_time
    
    # Estimate the full training time based on the subset time
    estimated_full_time = elapsed_time_subset / subsample_fraction
    
    # Format the estimated time in hours and minutes
    formatted_time = format_time(estimated_full_time)
    
    return formatted_time