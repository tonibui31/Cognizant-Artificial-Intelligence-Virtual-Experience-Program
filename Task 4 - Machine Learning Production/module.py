# 1. IMPORTING PACKAGES

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from datetime import datetime

# 2. DEFINE GLOBAL CONSTANTS
SALE_PATH = "sales.csv"
STOCK_PATH = "sensor_stock_levels.csv"
TEMP_PATH = "sensor_storage_temperature.csv"

# 3. ALGORITHM CODE

# Load Data
def data_loading(path):
    '''
    This function takes a path string to a CSV file and loads it into a Pandas DataFrame
    
    :param  path(optional):str, relative path of the CSV file
    
    :return df: pd.DataFrame
    '''
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# Convert time to standard datetime
def convert_to_datetime(data: pd.DataFrame = None, column: str = None):
    datetime_converted_df = data.copy()
    datetime_converted_df[column] = pd.to_datetime(datetime_converted_df[column], format='%Y-%m-%d %H:%M:%S')
    return datetime_converted_df

def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
    hourly_converted_df = data.copy()
    new_ts = hourly_converted_df[column].tolist()
    new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
    new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
    hourly_converted_df[column] = new_ts
    return hourly_converted_df

# Create target variable and predictor variables
def create_X_y(data: pd.DataFrame = None, column: str = None):
    '''
    This function takes in a Pandas DataFrame and splits the columns into a target column and a set of predictor variables,
    i.e. X & y. These two splits of the data will be used to train a supervised machine learning model.
    
    :param  data: pd.DataFrame, dataframe containing data for the model
    :param  target: str, target variable that you want to predict
    
    :return X: pd.DataFrame, y: pd.Series
    '''
    X = data.drop(columns = [column])
    y = data[column]
    
    return X, y

# Train algorithm
def train_algorithm_with_cross_validation(X : pd.DataFrame = None, y : pd.Series = None, K = 10, split = 0.75):
    '''
    This function takes the predictor and target variables and trains a Random Forest Regressor model accros K folds.
    Standard Scaler will also be applied. Using cross-validation, performance metrics (mean_absolute_error) 
    will be output for each fold during training
    
    :param      X: pd.DataFrame, predictor variables
    :param      y: pd.Series, target variable

    :return
    '''
    accuracy = [] # Create a list that will store the accuracies of each fold
    
    for fold in range(0, K): # Enter a loop to run K folds of cross-validation

        # Instantiate algorithm
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")
        
    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")
    
# 4. MAIN FUNCTION
def model_execute():
    '''
    This function executes the training pipeline of loading the prepared dataset from a CSV file,
    split the dataset, then training the machine learning model
    
    :param
    
    :return
    '''
    # Load the data
    df = data_loading() # Data pre-processing, feature engineering first, merge all the data into final CSV file, then use this function
    
    # Split the data into into predictors and target variables
    X, y = create_X_y(data = df)
    
    # Train the ML model
    train_algorithm_with_cross_validation(X, y, K, split)