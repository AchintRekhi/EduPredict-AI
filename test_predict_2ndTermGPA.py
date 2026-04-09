# test_predict_2ndTermGPA.py
# Predict Second Term GPA using the trained Neural Network model.

import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("model/gpa_regression_model.h5")
scaler = joblib.load("model/gpa_scaler.pkl")


def predict_second_term_gpa(input_dict):
    """Predict Second Term GPA using trained NN model and scaler.

    Args:
        input_dict: Dictionary with keys 'First_Term_Gpa' and 'High_School_Average_Mark'.

    Returns:
        Predicted Second Term GPA rounded to 2 decimal places.
    """
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]
    return round(prediction, 2)


if __name__ == "__main__":
    example_input = {
        "First_Term_Gpa": 4.0,
        "High_School_Average_Mark": 100
    }

    predicted_gpa = predict_second_term_gpa(example_input)
    print(f"🎯 Predicted Second Term GPA: {predicted_gpa}")
