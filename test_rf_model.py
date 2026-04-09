# test_rf_model.py

import pandas as pd
import joblib

# Load the pre-trained model and preprocessor
rf_model = joblib.load("model/rf_model.pkl")
preprocessor = joblib.load("model/rf_preprocessor.pkl")

# Set your optimal threshold
THRESHOLD = 0.43

# Sample student input (make sure keys match the training columns)
sample_input = {
    'First_Term_Gpa': 1.28,
    'Second_Term_Gpa': 1.42,
    'First_Language': 3,
    'Funding': 2,
    'School': 4,
    'FastTrack': 2,
    'Coop': 1,
    'Residency': 1,
    'Gender': 2,
    'Previous_Education': 1,
    'Age_Group': 3,
    'High_School_Average_Mark': 20.0,
    'Math_Score': 40,
    'English_Grade': 8
}

# Convert to DataFrame
input_df = pd.DataFrame([sample_input])

# Apply preprocessing
X_input = preprocessor.transform(input_df)

# Predict probability
probability = rf_model.predict_proba(X_input)[0][1]

# Apply threshold
prediction = int(probability > THRESHOLD)

# Output result
print(f"🎯 Predicted Persistence: {'Yes' if prediction == 1 else 'No'}")
print(f"🧮 Probability of Persistence: {probability:.4f}")
