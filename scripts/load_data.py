import pandas as pd



def load_data(file_path):
    
    """
    Load data from from a .csv file
    Args:
        file_path (str): Path to the csv file
    Returns:
        pd.DataFrame: Loaded data.
    """
    
    return pd.read_csv(file_path)
