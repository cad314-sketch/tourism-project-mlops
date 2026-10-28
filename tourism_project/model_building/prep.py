import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # Load the registered dataset
    data_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # Drop unnecessary identifier columns if present
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Define features and target
    target_col = "ProdTaken"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save local CSV copies for the next job artifact
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print("Data preparation completed and splits saved successfully.")

if __name__ == "__main__":
    prepare_data()
