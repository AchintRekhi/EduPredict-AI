# test_predict_3rdTermGPA.py
# Predict GPA Improvement using the trained Neural Network model.

import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("model/gpa_improvement_model.h5")
scaler = joblib.load("model/gpa_improvement_scaler.pkl")


def predict_gpa_improvement(input_dict):
    """Predict GPA Improvement using trained NN model.

    Args:
        input_dict: Dictionary with keys 'High_School_Average_Mark',
                    'First_Term_Gpa', and 'Second_Term_Gpa'.

    Returns:
        Predicted GPA improvement rounded to 2 decimal places.
    """
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]
    return round(prediction, 2)


if __name__ == "__main__":
    example_input = {
        "High_School_Average_Mark": 80,
        "First_Term_Gpa": 2.8,
        "Second_Term_Gpa": 2.9
    }

    predicted_improvement = predict_gpa_improvement(example_input)
    print(f"📈 Predicted GPA Improvement: {predicted_improvement}")
