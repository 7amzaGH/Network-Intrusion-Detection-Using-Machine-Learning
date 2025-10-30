"""
Prediction script for Network Intrusion Detection System
"""

import pandas as pd
import joblib
import sys
from utils import label_encode_features


def load_model_artifacts(model_dir):
    """Load saved model and preprocessing objects"""
    
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    selected_features = joblib.load(f"{model_dir}/selected_features.pkl")
    best_model = joblib.load(f"{model_dir}/decision_tree.pkl")
    
    return best_model, scaler, selected_features


def predict(data_path, model_dir):
    """
    Make predictions on new data
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with new data
    model_dir : str
        Path to saved model directory
        
    Returns:
    --------
    predictions : array
        Predicted classes (0=normal, 1=anomaly)
    """
    
    # Load data
    print(f" Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    # Load model artifacts
    print(f" Loading model from: {model_dir}")
    model, scaler, selected_features = load_model_artifacts(model_dir)
    
    # Preprocess
    print(" Preprocessing...")
    data_encoded = label_encode_features(data)
    
    if 'num_outbound_cmds' in data_encoded.columns:
        data_encoded.drop(['num_outbound_cmds'], axis=1, inplace=True)
    
    data_selected = data_encoded[selected_features]
    data_scaled = scaler.transform(data_selected)
    
    # Predict
    print(" Making predictions...")
    predictions = model.predict(data_scaled)
    
    # Convert to labels
    labels = ['normal' if p == 1 else 'anomaly' for p in predictions]
    
    return predictions, labels


def main():
    """Main prediction pipeline"""
    
    if len(sys.argv) < 3:
        print("Usage: python predict.py <data_path> <model_dir>")
        print("Example: python predict.py data/Test_data.csv saved_models/version_20241030_143022")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_dir = sys.argv[2]
    
    print("=" * 80)
    print("NETWORK INTRUSION DETECTION SYSTEM - PREDICTION")
    print("=" * 80)
    
    predictions, labels = predict(data_path, model_dir)
    
    # Display results
    print("\n Results:")
    print(f"Total samples: {len(predictions)}")
    print(f"Normal: {labels.count('normal')}")
    print(f"Anomaly: {labels.count('anomaly')}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'prediction': predictions,
        'label': labels
    })
    results_df.to_csv('predictions.csv', index=False)
    print("\n Predictions saved to: predictions.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
