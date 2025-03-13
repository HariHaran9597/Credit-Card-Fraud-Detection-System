# Credit Card Fraud Detection System

## Project Overview
This project implements a machine learning-based system for detecting fraudulent credit card transactions. It combines powerful ML models with an interactive dashboard and a REST API for real-time fraud detection.

## Technical Architecture

### Components
1. **Machine Learning Models**
   - Isolation Forest for anomaly detection
   - XGBoost for supervised classification
   - SMOTE for handling class imbalance

2. **FastAPI Backend**
   - RESTful API endpoints for real-time predictions
   - Supports both Isolation Forest and XGBoost models
   - Health check endpoint for monitoring

3. **Streamlit Dashboard**
   - Interactive data visualization
   - Real-time model evaluation
   - Feature importance analysis

## Model Performance

The system uses two different models for fraud detection:

1. **XGBoost Classifier**
   - Precision: ~0.95
   - Recall: ~0.92
   - F1-Score: ~0.93
   - Handles imbalanced data using SMOTE

2. **Isolation Forest**
   - Unsupervised anomaly detection
   - Complements XGBoost for detecting novel fraud patterns

## Installation & Setup

### Using Docker
```bash
# Build the Docker image
docker build -t fraud-detection .

# Run the container
docker run -p 8501:8501 -p 8000:8000 fraud-detection
```

### Manual Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the services:
   ```bash
   # Start the API server
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000

   # Start the dashboard
   streamlit run src/dashboard/app.py
   ```

## API Documentation

### Endpoints

1. **XGBoost Prediction**
   ```
   POST /predict/xgboost
   ```
   Returns fraud probability and classification

2. **Isolation Forest Prediction**
   ```
   POST /predict/isolation-forest
   ```
   Returns anomaly detection results

3. **Health Check**
   ```
   GET /health
   ```
   Monitors system status

## Dashboard Features

1. **Dataset Overview**
   - Total transaction count
   - Fraud rate analysis
   - Class distribution visualization

2. **Model Evaluation**
   - Real-time model training
   - Performance metrics comparison
   - Feature importance visualization

## Project Structure
```
├── Dataset/
│   └── creditcard.csv
├── src/
│   ├── api/
│   │   └── app.py
│   ├── dashboard/
│   │   └── app.py
│   ├── data/
│   │   └── preprocessing.py
│   └── models/
│       └── train.py
├── Dockerfile
└── requirements.txt
```

## Technologies Used
- Python 3.11
- FastAPI
- Streamlit
- XGBoost
- Scikit-learn
- Pandas & NumPy
- Docker

## Future Improvements
1. Add real-time monitoring dashboard
2. Implement model retraining pipeline
3. Add more advanced fraud detection algorithms
4. Enhance API security features

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.