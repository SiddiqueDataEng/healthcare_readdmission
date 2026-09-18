"""
Quick Data Generator for Demo Purposes
Creates sample data when main data files are missing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path


def generate_sample_data():
    """Generate minimal sample data for demo purposes"""
    
    # Create directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Generate sample patients
    np.random.seed(42)
    n_patients = 100
    
    patients_data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.randint(18, 90, n_patients),
        'gender': np.random.choice(['Male', 'Female'], n_patients),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_patients),
        'insurance_type': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Uninsured'], n_patients),
        'has_hypertension': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
        'has_diabetes': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'has_heart_disease': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'bmi': np.random.normal(28, 5, n_patients),
        'last_lab_value': np.random.normal(100, 20, n_patients),
        'readmitted_30d': np.random.choice([0, 1], n_patients, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(patients_data)
    
    # Add some missing values for demo
    missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_indices, 'last_lab_value'] = np.nan
    
    # Add some duplicates for demo
    duplicate_rows = df.sample(n=5)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Save files
    df.to_csv('data/processed/full_dataset.csv', index=False)
    
    # Create train/test split
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    train_df.to_csv('data/processed/train_data.csv', index=False)
    test_df.to_csv('data/processed/test_data.csv', index=False)
    
    return df


def check_data_exists():
    """Check if processed data files exist"""
    files = [
        'data/processed/full_dataset.csv',
        'data/processed/train_data.csv',
        'data/processed/test_data.csv'
    ]
    
    return all(os.path.exists(f) for f in files)


if __name__ == "__main__":
    print("Generating sample data...")
    df = generate_sample_data()
    print(f"✅ Generated sample data with {len(df)} rows")
    print("Files created:")
    print("  - data/processed/full_dataset.csv")
    print("  - data/processed/train_data.csv") 
    print("  - data/processed/test_data.csv")