# test_2nd_gpa_rf.py
# Predict Second Term GPA using the trained Random Forest model.

import joblib
import numpy as np

model = joblib.load("model/rf_gpa_model.pkl")

# Example input: [First_Term_Gpa, High_School_Average_Mark]
input_data = np.array([[3.2, 85.0]])

predicted_gpa = model.predict(input_data)
print(f"📈 Predicted Second Term GPA: {predicted_gpa[0]:.2f}")
