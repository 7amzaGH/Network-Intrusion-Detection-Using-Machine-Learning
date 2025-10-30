"""
Utility functions for Network Intrusion Detection System
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encode_features(df):
    """
    Apply label encoding to all object-type columns
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
        
    Returns:
    --------
    df : pandas DataFrame
        Encoded dataframe
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
    return df


def load_and_preprocess_data(train_path, test_path):
    """
    Load and preprocess training and test data
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str
        Path to test CSV file
        
    Returns:
    --------
    train, test : pandas DataFrames
        Preprocessed data
    """
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Encode categorical features
    train = label_encode_features(train)
    test = label_encode_features(test)
    
    # Remove unnecessary columns
    if 'num_outbound_cmds' in train.columns:
        train.drop(['num_outbound_cmds'], axis=1, inplace=True)
        test.drop(['num_outbound_cmds'], axis=1, inplace=True)
    
    return train, test
