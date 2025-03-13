from setuptools import setup, find_packages

setup(
    name="fraud_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "imbalanced-learn",
        "matplotlib",
        "seaborn",
        "fastapi",
        "uvicorn",
        "pydantic",
        "streamlit"
    ]
)