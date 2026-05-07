import pandas as pd
import numpy as np
import os

def clean_healthcare_data(input_path, output_path):
    """
    Cleans the healthcare assessment dataset.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Handling Missing Values
    # Categorical
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['Medical Condition'] = df['Medical Condition'].fillna('Unknown')
    
    # Numeric (Median is safer than mean for potentially skewed health data)
    numeric_cols = ['Glucose', 'Blood Pressure', 'BMI', 'Oxygen Saturation', 
                    'Cholesterol', 'Triglycerides', 'HbA1c', 'Diet Score', 
                    'Stress Level', 'Sleep Hours', 'Age']
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 2. Correcting Data Types
    df['Age'] = df['Age'].astype(int)

    # 3. Handling Logical Inconsistencies
    # Physical activity cannot be negative
    df['Physical Activity'] = df['Physical Activity'].clip(lower=0)

    # 4. Removing Duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows.")

    # 5. Dropping Noise Columns
    cols_to_drop = ['noise_col', 'random_notes']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 6. Ensure Output Directory Exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 7. Saving the Cleaned Data
    df.to_csv(output_path, index=False)
    print(f"Success! Cleaned data saved to {output_path}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    # Get absolute paths relative to this script or project root
    # For simplicity, we'll assume we run this from the project root
    RAW_DATA = "data/raw/dirty_v3_path.csv"
    PROCESSED_DATA = "data/processed/cleaned_healthcare_data.csv"
    
    if os.path.exists(RAW_DATA):
        clean_healthcare_data(RAW_DATA, PROCESSED_DATA)
    else:
        print(f"Error: {RAW_DATA} not found. Please run from project root.")
