# train_gpa_regression_rf.py
# Random Forest Model Training Script to Predict Second Term GPA
# This script trains a Random Forest model to predict the second term GPA of students based on their first term GPA and high school average mark.


import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the cleaned dataset
df = pd.read_csv("data/Student_Data.csv")
df = df.dropna(subset=["Second_Term_Gpa"])

# Select features and target
X = df[["First_Term_Gpa", "High_School_Average_Mark"]]
y = df["Second_Term_Gpa"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Define parameter grid for tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Initialize and tune Random Forest using GridSearchCV
base_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best model from tuning
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, "model/rf_gpa_model.pkl")

# Predict and evaluate
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print evaluation
print("🌲 Tuned Random Forest GPA Model Evaluation")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²    : {r2:.4f}")

# Plot predicted vs actual GPA
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Second Term GPA")
plt.ylabel("Predicted Second Term GPA")
plt.title("Tuned RF Model: Actual vs Predicted Second Term GPA")
plt.grid(True)
plt.tight_layout()
plt.savefig("model/rf_gpa_plot_tuned.png")
plt.show()
