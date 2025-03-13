import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.models.train import FraudDetectionModel
from src.data.preprocessing import DataPreprocessor

def main():
    # Load the credit card dataset
    print("Loading dataset...")
    data = pd.read_csv('Dataset/creditcard.csv')
    
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize preprocessor and models
    preprocessor = DataPreprocessor()
    model = FraudDetectionModel()
    
    # Preprocess the data
    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    
    # Train models
    print("Training Isolation Forest model...")
    model.train_isolation_forest(X_train_processed)
    
    print("Training XGBoost model...")
    model.train_xgboost(X_train_balanced, y_train_balanced)
    
    # Evaluate models
    print("Evaluating models...")
    evaluation_results = model.evaluate_models(X_test_processed, y_test)
    
    # Print evaluation results
    print("\nIsolation Forest Results:")
    print(f"Precision: {evaluation_results['isolation_forest']['weighted avg']['precision']:.4f}")
    print(f"Recall: {evaluation_results['isolation_forest']['weighted avg']['recall']:.4f}")
    print(f"F1-score: {evaluation_results['isolation_forest']['weighted avg']['f1-score']:.4f}")
    
    print("\nXGBoost Results:")
    print(f"Precision: {evaluation_results['xgboost']['weighted avg']['precision']:.4f}")
    print(f"Recall: {evaluation_results['xgboost']['weighted avg']['recall']:.4f}")
    print(f"F1-score: {evaluation_results['xgboost']['weighted avg']['f1-score']:.4f}")
    
    # Save the trained models
    print("\nSaving models...")
    model.save_models('src/models')
    print("Models saved successfully!")

if __name__ == "__main__":
    main()