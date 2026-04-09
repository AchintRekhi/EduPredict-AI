# 🎓 Student GPA & Persistence Prediction System

Predicting student academic outcomes using Neural Networks and Random Forest models trained on real institutional data.

## 📌 What This Project Does

| Prediction Task | Type | Models |
|----------------|------|--------|
| **First Year Persistence** | Classification | Neural Network, Random Forest |
| **Second Term GPA** | Regression | Neural Network, Random Forest (GridSearchCV) |
| **GPA Improvement** (2nd − 1st) | Regression | Neural Network |

---

## 📁 Project Structure

```
.
├── data/                              # Cleaned dataset
│   └── Student_Data.csv
├── model/                             # Trained models & scalers
│   ├── trained_model.h5               # Persistence NN
│   ├── rf_model.pkl                   # Persistence RF
│   ├── gpa_regression_model.h5        # GPA Regression NN
│   ├── rf_gpa_model.pkl               # GPA Regression RF (tuned)
│   ├── gpa_improvement_model.h5       # GPA Improvement NN
│   └── *.pkl                          # Scalers / preprocessors
├── webapp/                            # Flask web application
│   ├── app.py                         # Backend API
│   ├── templates/index.html           # Frontend UI
│   └── static/                        # CSS & JS
├── notebooks/                         # Jupyter development notebooks
│   └── Group-2.ipynb
├── docs/                              # Documentation & references
│   ├── input_fields_GPA_models.txt
│   ├── persistence_model_input_fields.txt
│   ├── COMP258F23GroupProject.pdf
│   └── Description of Variables.pdf
├── train_model.py                     # Train persistence NN
├── train_rf_model.py                  # Train persistence RF
├── train_gpa_regression_nn.py         # Train GPA regression NN
├── train_gpa_regression_rf.py         # Train GPA regression RF
├── train_gpa_improvement_nn.py        # Train GPA improvement NN
├── test_predict_2ndTermGPA.py         # Test GPA prediction (NN)
├── test_predict_3rdTermGPA.py         # Test GPA improvement prediction
├── test_rf_model.py                   # Test persistence prediction (RF)
├── test_2nd_gpa_rf.py                 # Test GPA prediction (RF)
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 📊 Dataset

- **1,437 student records** with 14 features
- **Numeric**: First/Second Term GPA, High School Average, Math Score
- **Ordinal**: English Grade, Age Group, Previous Education
- **Categorical**: Language, Funding, Gender, FastTrack, Co-op, Residency
- **Target**: FirstYearPersistence (binary)

---

## 🛠️ Setup & Installation

```bash
pip install -r requirements.txt
```

### Train a model
```bash
python train_gpa_regression_nn.py
```

### Run the web app
```bash
cd webapp && python app.py
```
Then open **http://localhost:5000** in your browser.

---

## 📏 Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, PR-AUC
- **Regression**: MAE, RMSE, R² Score

---

## 👥 Team

COMP-258 Neural Networks — Group 2 Final Project
