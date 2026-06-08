import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(csv_path, target_column):

    df = pd.read_csv(csv_path)

    # Remove non-numeric cols (dates etc.)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True).dt.year
            except:
                df = df.drop(columns=[col])

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    if target_column not in df.columns:
        raise ValueError(f"{target_column} missing in CSV")

    # Extract X, y
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Normalize X and y
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    X_val   = X_scaler.transform(X_val)
    X_test  = X_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_scaler, y_scaler
    )
