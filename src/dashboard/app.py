import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.preprocessing import DataPreprocessor
from src.models.train import FraudDetectionModel

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")

# Initialize components
preprocessor = DataPreprocessor()
model = FraudDetectionModel()

def load_and_process_data():
    """Load and preprocess the credit card transaction data."""
    data = preprocessor.load_data('Dataset/creditcard.csv')
    X, y = preprocessor.preprocess_data(data)
    return data, X, y

def plot_class_distribution(y):
    """Plot the distribution of fraudulent vs non-fraudulent transactions."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Fraud vs Normal Transactions')
    plt.xlabel('Class (0: Normal, 1: Fraud)')
    plt.ylabel('Count')
    return plt

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the XGBoost model."""
    importance = model.xgboost.feature_importances_
    feat_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feat_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    return plt

def main():
    st.title('Credit Card Fraud Detection Dashboard')
    
    # Load data
    with st.spinner('Loading data...'):
        data, X, y = load_and_process_data()
    
    # Display basic statistics
    st.header('Dataset Overview')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Transactions', len(data))
    with col2:
        st.metric('Fraudulent Transactions', sum(y == 1))
    with col3:
        fraud_rate = (sum(y == 1) / len(y)) * 100
        st.metric('Fraud Rate', f'{fraud_rate:.2f}%')
    
    # Class distribution plot
    st.header('Class Distribution')
    st.pyplot(plot_class_distribution(y))
    
    # Model performance metrics
    st.header('Model Performance')
    if st.button('Evaluate Models'):
        with st.spinner('Evaluating models...'):
            # Split data
            X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
            
            # Train models
            model.train_isolation_forest(X_train)
            model.train_xgboost(X_train, y_train)
            
            # Get evaluation metrics
            metrics = model.evaluate_models(X_test, y_test)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Isolation Forest Metrics')
                st.json(metrics['isolation_forest'])
            with col2:
                st.subheader('XGBoost Metrics')
                st.json(metrics['xgboost'])
    
    # Feature importance plot
    st.header('Feature Importance (XGBoost)')
    if hasattr(model.xgboost, 'feature_importances_'):
        st.pyplot(plot_feature_importance(model, preprocessor.get_feature_names()))
    else:
        st.info('Train the models first to see feature importance')

if __name__ == '__main__':
    main()