import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.preprocessing import prepare_data, SPECIES_NAMES

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def get_classifiers(random_state=42):
    """
    Returns a dictionary of candidate classifier models.
    """
    return {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=200),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
    }

def train_and_evaluate_all():
    """
    Trains all candidate classifiers, evaluates performance metrics, selects the best model,
    and saves the best model and scaler to the models directory.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = prepare_data()
    classifiers = get_classifiers()
    
    results = []
    trained_models = {}
    evaluation_details = {}
    
    for name, model in classifiers.items():
        # Train model (Decision Tree & Random Forest don't strictly require scaling, but using scaled input preserves pipeline uniformity)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=SPECIES_NAMES, output_dict=True)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        trained_models[name] = model
        evaluation_details[name] = {
            'confusion_matrix': cm,
            'report': report,
            'y_pred': y_pred
        }
    
    results_df = pd.DataFrame(results).sort_values(by=['F1-Score', 'Accuracy'], ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, 'iris_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    meta_info = {
        'best_model_name': best_model_name,
        'model_path': model_path,
        'scaler_path': scaler_path
    }
    joblib.dump(meta_info, os.path.join(MODELS_DIR, 'model_meta.pkl'))
    
    print("=" * 60)
    print("MODEL TRAINING & EVALUATION RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("-" * 60)
    print(f"Best Performing Model Selected: {best_model_name}")
    print(f"Saved Model to: {model_path}")
    print(f"Saved Scaler to: {scaler_path}")
    print("=" * 60)
    
    return results_df, trained_models, evaluation_details, meta_info

if __name__ == '__main__':
    train_and_evaluate_all()
