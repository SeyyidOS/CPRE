import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(data, test_size=0.2, val_size=0.1, target_column='Rating'):
    y = data['Rating']
    X = data.drop(columns=["Rating"])
    
    num_bins = 10  
    y_binned = pd.qcut(y, q=num_bins, labels=False)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y_binned, random_state=42
    )

    num_bins = 10  
    y_binned = pd.qcut(y_train_val, q=num_bins, labels=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_binned, random_state=42
    )

    return X_train, y_train, X_val, y_val, X_test, y_test