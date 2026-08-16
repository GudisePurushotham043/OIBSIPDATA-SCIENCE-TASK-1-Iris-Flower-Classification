import os
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import FEATURE_NAMES, SPECIES_NAMES

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def load_saved_model_and_scaler():
    """
    Loads the saved best model, scaler, and metadata.
    """
    model_path = os.path.join(MODELS_DIR, 'iris_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    meta_path = os.path.join(MODELS_DIR, 'model_meta.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Model or scaler file not found. Please run training first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else {'best_model_name': 'Best Model'}
    
    return model, scaler, meta

def validate_inputs(sepal_length, sepal_width, petal_length, petal_width):
    """
    Validates measurement inputs to ensure they are positive numeric values.
    """
    inputs = [sepal_length, sepal_width, petal_length, petal_width]
    names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    for val, name in zip(inputs, names):
        if val is None:
            raise ValueError(f"{name} is missing. Please provide a valid value.")
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric input for {name}: {val}")
            
        if val_float <= 0:
            raise ValueError(f"{name} must be a positive value greater than 0.")
            
    return float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)

def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predicts the Iris species for given sepal and petal dimensions.
    
    Returns:
        dict: Containing species, species_title, confidence, probabilities, model_used, and inputs.
    """
    sl, sw, pl, pw = validate_inputs(sepal_length, sepal_width, petal_length, petal_width)
    
    model, scaler, meta = load_saved_model_and_scaler()
    
    # Feature array in exact order expected by DataFrame / Scaler
    input_features = np.array([[sl, sw, pl, pw]])
    input_df = pd.DataFrame(input_features, columns=FEATURE_NAMES)
    
    scaled_features = scaler.transform(input_df)
    
    prediction_idx = model.predict(scaled_features)[0]
    predicted_species = SPECIES_NAMES[prediction_idx]
    
    probabilities = {}
    confidence = 100.0
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(scaled_features)[0]
        confidence = float(np.max(proba) * 100)
        for idx, species_name in enumerate(SPECIES_NAMES):
            probabilities[species_name.capitalize()] = round(float(proba[idx] * 100), 2)
    else:
        for idx, species_name in enumerate(SPECIES_NAMES):
            probabilities[species_name.capitalize()] = 100.0 if idx == prediction_idx else 0.0

    return {
        'species': predicted_species,
        'species_title': predicted_species.capitalize(),
        'confidence': round(confidence, 2),
        'probabilities': probabilities,
        'model_used': meta.get('best_model_name', 'Trained Classifier'),
        'inputs': {
            'Sepal Length (cm)': sl,
            'Sepal Width (cm)': sw,
            'Petal Length (cm)': pl,
            'Petal Width (cm)': pw
        }
    }

if __name__ == '__main__':
    # Test sample prediction
    sample_res = predict_species(5.1, 3.5, 1.4, 0.2)
    print("Sample Prediction Result:", sample_res)
