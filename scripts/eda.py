import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Task 1: Overview of the Data
def overview_of_data(df):
    print("--- Overview of the Dataset ---")
    print("Number of Rows:", df.shape[0])
    print("Number of Columns:", df.shape[1])
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 Rows:")
    print(df.head())

# Task 2: Summary Statistics
def summary_statistics(df):
    print("--- Summary Statistics ---")
    print(df.describe(include='all'))

# Task 3: Distribution of Numerical Features
def plot_numerical_distributions(df):
    print("--- Numerical Feature Distributions ---")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

# Task 4: Distribution of Categorical Features
def plot_categorical_distributions(df):
    print("--- Categorical Feature Distributions ---")
    categorical_cols = [ 'CurrencyCode', 'ProviderId',  'ProductCategory','ChannelId']
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, palette='Set2')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

# Task 5: Correlation Analysis
def correlation_analysis(df):
    print("--- Correlation Analysis ---")
    numerical_cols = df.select_dtypes(include=[np.number])
    correlation_matrix = numerical_cols.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

# Task 6: Identifying Missing Values
def missing_values_analysis(df):
    print("--- Missing Values Analysis ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

# Task 7: Outlier Detection
def outlier_detection(df):
    print("--- Outlier Detection ---")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=col, palette='Set1')
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)
        plt.show()
