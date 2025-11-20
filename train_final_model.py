# ==============================================================================
# IMPORT LIBRARIES
# ==============================================================================
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("--- Model Training Script Started ---")

# ==============================================================================
# 1. LOAD THE CLEANED DATA
# ==============================================================================
# This is the output from Step 3.
try:
    df = pd.read_csv('data/cleaned_appointments.csv')
    print(f"Successfully loaded 'cleaned_appointments.csv' with {df.shape[0]} rows.")
except FileNotFoundError:
    print("Error: 'data/cleaned_appointments.csv' not found. Please run Step 3 first.")
    exit()

# ==============================================================================
# 2. DEFINE FEATURES (X) AND TARGET (y)
# ==============================================================================
# These are the 8 features from your team's report.
# The order here doesn't matter for the pipeline, but consistency is good practice.
feature_names = [
    'age',
    'lead_time_days',
    'appointment_dow',
    'sms_received',
    'total_conditions',
    'gender',
    'hypertension',
    'has_chronic_condition'
]

target_name = 'no_show' # This is the column we want to predict (0 or 1)

X = df[feature_names]
y = df[target_name]

print(f"Features (X) and target (y) have been defined. Using {len(feature_names)} features.")

# ==============================================================================
# 3. DEFINE AND TRAIN THE PIPELINE
# ==============================================================================
# A pipeline bundles preprocessing and modeling steps.
# This ensures that when we make predictions, the new data goes through the exact same steps.
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])
# Note: `class_weight='balanced'` is helpful if you have more 'shows' than 'no-shows'.

# Split the data into a training set and a testing set for evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Note: `stratify=y` ensures the proportion of no-shows is the same in train and test sets.

print("\nTraining the model pipeline... (This may take a moment)")
pipeline.fit(X_train, y_train)
print("Training complete.")

# ==============================================================================
# 4. EVALUATE THE MODEL
# ==============================================================================
# Check the model's performance on the unseen test data.
accuracy = pipeline.score(X_test, y_test)
print(f"\nModel Accuracy on Test Set: {accuracy:.2%}")

# For a deeper look, generate a classification report (precision, recall, f1-score)
y_pred = pipeline.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=['Will Show (0)', 'No-Show (1)']))

# ==============================================================================
# 5. SAVE THE FINAL PIPELINE
# ==============================================================================
# This is the final, single artifact that your Streamlit app will load.
FINAL_PIPELINE_PATH = 'model/final_model_pipeline.pkl'
joblib.dump(pipeline, FINAL_PIPELINE_PATH, compress=3)

print(f"\n✅ SUCCESS! Final pipeline saved to: '{FINAL_PIPELINE_PATH}'")
print("--- Model Training Script Finished ---")