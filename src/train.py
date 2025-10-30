"""
Training script for Network Intrusion Detection System
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from utils import load_and_preprocess_data

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def select_features(X, y, n_features=10):
    """Select top features using RFE"""
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    rfe = RFE(rfc, n_features_to_select=n_features)
    rfe.fit(X, y)
    
    feature_map = [(i, v) for i, v in zip(rfe.get_support(), X.columns)]
    selected_features = [v for i, v in feature_map if i]
    
    return selected_features


def train_models(X_train, y_train):
    """Train multiple models"""
    
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=4),
        'Logistic Regression': LogisticRegression(max_iter=1200000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, max_features=5, random_state=RANDOM_STATE)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models


def save_models(models, scaler, selected_features, save_dir='saved_models'):
    """Save all trained models and artifacts"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = f"{save_dir}/version_{timestamp}"
    os.makedirs(version_dir, exist_ok=True)
    
    # Save models
    for name, model in models.items():
        safe_name = name.replace(' ', '_').lower()
        joblib.dump(model, f"{version_dir}/{safe_name}.pkl")
    
    # Save preprocessing objects
    joblib.dump(scaler, f"{version_dir}/scaler.pkl")
    joblib.dump(selected_features, f"{version_dir}/selected_features.pkl")
    
    print(f" Models saved to: {version_dir}/")
    return version_dir


def main():
    """Main training pipeline"""
    
    print("=" * 80)
    print("NETWORK INTRUSION DETECTION SYSTEM - TRAINING")
    print("=" * 80)
    
    # 1. Load and preprocess data
    print("\n Loading data...")
    train, test = load_and_preprocess_data(
        'data/Train_data.csv',
        'data/Test_data.csv'
    )
    
    # 2. Separate features and target
    X_train = train.drop(['class'], axis=1)
    y_train = train['class']
    
    # 3. Feature selection
    print("\n Selecting features...")
    selected_features = select_features(X_train, y_train, n_features=10)
    X_train = X_train[selected_features]
    test = test[selected_features]
    
    # 4. Standardization
    print("\n Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 5. Train-test split
    x_train, x_test, y_train_split, y_test = train_test_split(
        X_train_scaled, y_train,
        train_size=0.70,
        random_state=RANDOM_STATE
    )
    
    # 6. Train models
    print("\n Training models...")
    trained_models = train_models(x_train, y_train_split)
    
    # 7. Evaluate
    print("\n Model Performance:")
    for name, model in trained_models.items():
        score = model.score(x_test, y_test)
        print(f"  {name}: {score*100:.2f}%")
    
    # 8. Save models
    print("\n Saving models...")
    save_models(trained_models, scaler, selected_features)
    
    print("\n Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
