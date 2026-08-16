import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

SPECIES_NAMES = ['setosa', 'versicolor', 'virginica']

def load_iris_data():
    """
    Loads the Iris dataset from scikit-learn and returns it as a pandas DataFrame.
    Returns:
        df (pd.DataFrame): DataFrame with features, target (int), and species_name (str).
        target_names (list): List of species names.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species_name'] = df['target'].apply(lambda x: iris.target_names[x])
    return df, list(iris.target_names)

def prepare_data(df=None, test_size=0.2, random_state=42):
    """
    Prepares features X and target y, splits them into train and test sets,
    and fits a StandardScaler on the training set.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
    """
    if df is None:
        df, _ = load_iris_data()
    
    X = df[FEATURE_NAMES]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test
