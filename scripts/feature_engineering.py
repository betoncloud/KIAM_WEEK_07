import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


# Task 1: Create Aggregate Features

def create_aggregate_features(df, id_column, amount_column):
    print("--- Creating Aggregate Features ---")
    aggregate_features = df.groupby(id_column).agg(
        total_transaction_amount=(amount_column, 'sum'),
        average_transaction_amount=(amount_column, 'mean'),
        transaction_count=(amount_column, 'count'),
        std_transaction_amount=(amount_column, 'std')
    ).reset_index()
    print("Aggregate features created successfully!")
    return aggregate_features
# Task 2: Extract Features
def extract_transaction_features(df, datetime_column):
    print("--- Extracting Date and Time Features ---")
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['transaction_hour'] = df[datetime_column].dt.hour
    df['transaction_day'] = df[datetime_column].dt.day
    df['transaction_month'] = df[datetime_column].dt.month
    df['transaction_year'] = df[datetime_column].dt.year
    print("Date and time features extracted successfully!")
    return df

# Task 3: Encode Categorical Variables
def encode_categorical_variables(df, categorical_columns):
    print("--- Encoding Categorical Variables ---")
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    print("One-hot encoding applied successfully!")

    # Label Encoding
    #label_encoders = {}
    for col in categorical_columns:
        df[col + 'encoded'] = df[col].str.extract(r'_(\d+)$').astype(int)
    print("Label encoding applied successfully!")
    return df

# Task 4: Handle Missing Values
def handle_missing_values(df, strategy='mean', columns=None):
    print("--- Handling Missing Values ---")
    imputer = SimpleImputer(strategy=strategy)
    if columns is None:
        columns = df.columns
    df[columns] = imputer.fit_transform(df[columns])
    print(f"Missing values imputed using {strategy} strategy!")
    return df

# Task 5: Normalize/Standardize Numerical Features
def scale_numerical_features(df, numerical_columns, method='standardize'):
    print("--- Scaling Numerical Features ---")
    scaler = StandardScaler() if method == 'standardize' else MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    print(f"Numerical features scaled using {method} method!")
    return df