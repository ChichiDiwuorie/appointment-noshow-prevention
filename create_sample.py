# create_sample.py

import pandas as pd
import joblib

print("--- Starting Sample Data Creation Script ---")

# --- 1. Load the FINAL Cleaned Data ---
# This is the same data your model was trained on.
try:
    df_clean = pd.read_csv('data/cleaned_appointments.csv')
    print("Successfully loaded 'data/cleaned_appointments.csv'.")
except FileNotFoundError:
    print("Error: 'data/cleaned_appointments.csv' not found. Please run Step 3 first.")
    exit()

# --- 2. Load the Trained Pipeline to get the exact feature list ---
# This is a robust way to ensure we have the right columns.
try:
    pipeline = joblib.load('model/final_model_pipeline.pkl')
    required_features = pipeline.feature_names_in_
    print(f"Model requires these {len(required_features)} columns: {list(required_features)}")
except FileNotFoundError:
    print("Error: 'model/final_model_pipeline.pkl' not found. Please run Step 4 & 5 first.")
    exit()

# --- 3. Create the Sample ---
# Take the first 100 rows from the clean data.
sample_df = df_clean.head(100)

# --- 4. Select ONLY the feature columns the model needs ---
# The target column ('no_show') is automatically excluded.
# This is the most critical step.
final_sample_df = sample_df[required_features]

# --- 5. Save the New Sample File ---
# This will overwrite your old, incorrect sample file.
output_path = 'data/appointment_sample_data.csv'
final_sample_df.to_csv(output_path, index=False)

print("\n--- Verification ---")
print(f"First 5 rows of the new sample file to be saved to '{output_path}':")
print(final_sample_df.head())
print(f"\n✅ SUCCESS! New sample file created at '{output_path}'.")