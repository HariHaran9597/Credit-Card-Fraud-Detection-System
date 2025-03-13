import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

class FraudDetectionModel:
    def __init__(self, contamination=0.02, scale_pos_weight=50):
        self.isolation_forest = IsolationForest(
            random_state=42,
            contamination=contamination,  # Increased contamination for better fraud detection
            n_estimators=200  # Increased number of trees for better model stability
        )
        self.xgboost = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            use_label_encoder=False,
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,  # Increased weight for better handling of class imbalance
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            random_state=42
        )
        self.performance_metrics = {}
        self.processed_transactions = 0

    def train_isolation_forest(self, X_train):
        """Train the Isolation Forest model for anomaly detection."""
        self.isolation_forest.fit(X_train)

    def train_xgboost(self, X_train, y_train):
        """Train the XGBoost model for classification."""
        self.xgboost.fit(X_train, y_train)

    def predict_isolation_forest(self, X):
        """Predict using Isolation Forest model."""
        # Convert predictions to binary (1 for inliers, 0 for outliers)
        return np.where(self.isolation_forest.predict(X) == 1, 0, 1)

    def predict_xgboost(self, X):
        """Predict using XGBoost model."""
        return self.xgboost.predict(X)

    def predict_xgboost_proba(self, X):
        """Get probability predictions from XGBoost model."""
        return self.xgboost.predict_proba(X)

    def evaluate_models(self, X_test, y_test, track_metrics=True):
        """Evaluate both models and return their performance metrics."""
        # Isolation Forest predictions
        if_preds = self.predict_isolation_forest(X_test)
        if_report = classification_report(y_test, if_preds, output_dict=True)

        # XGBoost predictions
        xgb_preds = self.predict_xgboost(X_test)
        xgb_report = classification_report(y_test, xgb_preds, output_dict=True)

        # Calculate confusion matrices
        if_cm = confusion_matrix(y_test, if_preds)
        xgb_cm = confusion_matrix(y_test, xgb_preds)

        # Track performance metrics if requested
        if track_metrics:
            self.processed_transactions += len(X_test)
            self.performance_metrics = {
                'total_processed': self.processed_transactions,
                'false_negative_rate': {
                    'isolation_forest': if_cm[1][0] / (if_cm[1][0] + if_cm[1][1]),
                    'xgboost': xgb_cm[1][0] / (xgb_cm[1][0] + xgb_cm[1][1])
                },
                'precision': {
                    'isolation_forest': if_report['1']['precision'],
                    'xgboost': xgb_report['1']['precision']
                },
                'recall': {
                    'isolation_forest': if_report['1']['recall'],
                    'xgboost': xgb_report['1']['recall']
                },
                'f1_score': {
                    'isolation_forest': if_report['1']['f1-score'],
                    'xgboost': xgb_report['1']['f1-score']
                }
            }

        return {
            'isolation_forest': if_report,
            'xgboost': xgb_report,
            'confusion_matrix': {
                'isolation_forest': if_cm.tolist(),
                'xgboost': xgb_cm.tolist()
            }
        }

    def save_models(self, base_path):
        """Save both models to disk."""
        joblib.dump(self.isolation_forest, f'{base_path}/isolation_forest.joblib')
        joblib.dump(self.xgboost, f'{base_path}/xgboost.joblib')

    def load_models(self, base_path):
        """Load both models from disk."""    
        self.isolation_forest = joblib.load(f'{base_path}/isolation_forest.joblib')
        self.xgboost = joblib.load(f'{base_path}/xgboost.joblib')

    def test_edge_cases(self, edge_cases_X, edge_cases_y):
        """Test model performance on specific edge cases.
        
        Args:
            edge_cases_X: DataFrame containing edge case scenarios
            edge_cases_y: True labels for edge cases
        
        Returns:
            Dictionary containing performance metrics for edge cases
        """
        # Evaluate models on edge cases
        edge_case_metrics = self.evaluate_models(edge_cases_X, edge_cases_y, track_metrics=False)
        
        # Add specific edge case analysis
        xgb_proba = self.predict_xgboost_proba(edge_cases_X)
        high_confidence_errors = sum((xgb_proba[:, 1] > 0.9) & (edge_cases_y == 0))
        
        edge_case_metrics['edge_case_analysis'] = {
            'high_confidence_false_positives': int(high_confidence_errors),
            'total_edge_cases': len(edge_cases_X)
        }
        
        return edge_case_metrics

    def get_performance_summary(self):
        """Get a summary of model performance metrics.
        
        Returns:
            Dictionary containing performance metrics and transaction statistics
        """
        return self.performance_metrics