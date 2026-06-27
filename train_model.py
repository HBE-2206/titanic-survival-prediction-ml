import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def load_and_train():
    """Load dataset, preprocess data, train model and save it."""
    # Load the dataset from the CSV file
    try:
        df = pd.read_csv('titanic_clean.csv')
    except FileNotFoundError:
        print("Error: titanic_clean.csv not found!")
        return

    # Standardize column names by removing spaces and capitalizing
    df.columns = [col.strip().capitalize() for col in df.columns]
    print(f"Columns found: {df.columns.tolist()}")

    # Convert mixed text-numeric columns to clean float numbers
    cols_to_fix = ['Age', 'Fare', 'Agecat', 'Farecat']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace missing Age and Fare values using backup category columns
    if ('Age' not in df.columns or df['Age'].isnull().all()) and 'Agecat' in df.columns:
        df['Age'] = df['Agecat']
    
    if ('Fare' not in df.columns or df['Fare'].isnull().all()) and 'Farecat' in df.columns:
        df['Fare'] = df['Farecat']

    # Fill remaining empty values with the median for numerical columns
    required = ['Pclass', 'Sex', 'Age', 'Sibsp', 'Parch', 'Fare', 'Embarked']
    for col in required:
        if col not in df.columns:
            print(f"Adding missing column: {col}")
            df[col] = 0
        df[col] = df[col].fillna(df[col].median() if col in ['Age', 'Fare'] else 0)

    # Convert categorical text (Sex/Embarked) into numerical values
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'].astype(str))
    df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
    
    # Define the features (X) and the target variable (y)
    X = df[required]
    target_col = 'Survived' if 'Survived' in df.columns else 'survived'
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found!")
        return
        
    y = df[target_col]
    
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    
    # Initialize and train the Decision Tree model
    model = DecisionTreeClassifier(max_depth=5, random_state=30)
    model.fit(X_train, y_train)
    
    # Make predictions and calculate the model's accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Model Accuracy: {acc:.2%}")
    print("-" * 30)

    # Save the trained model to a file for later use
    with open('titanic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully as titanic_model.pkl")


# Execute the training function
if __name__ == "__main__":
    load_and_train()