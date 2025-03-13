import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def load_data(self, file_path):
        """Load the credit card transaction dataset."""
        return pd.read_csv(file_path)

    def preprocess_data(self, data):
        """Preprocess the data by scaling numerical features."""
        # Separate features and target
        X = data.drop('Class', axis=1)
        y = data['Class']

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled, y

    def split_data(self, X, y, test_size=0.2):
        """Split the data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance."""
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    def get_feature_names(self):
        """Return the feature names after preprocessing."""
        return self.scaler.feature_names_in_

    def transform_new_data(self, data):
        """Transform new data using the fitted scaler."""
        return self.scaler.transform(data)

    def fit_transform(self, X):
        """Fit the scaler and transform the data."""
        return self.scaler.fit_transform(X)

    def transform(self, X):
        """Transform data using the fitted scaler."""
        return self.scaler.transform(X)