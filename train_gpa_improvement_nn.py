# train_gpa_improvement_nn.py
# Neural Network Model to Predict GPA Improvement (Second Term − First Term)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Student_Data.csv")
df = df.dropna(subset=["First_Term_Gpa", "Second_Term_Gpa", "High_School_Average_Mark"])

# Create target: GPA Improvement
df["GPA_Improvement"] = df["Second_Term_Gpa"] - df["First_Term_Gpa"]

# Features and target
# NOTE: Second_Term_Gpa is included as an input feature here. Since the target
# (GPA_Improvement) is derived from it, there is potential data leakage.
# This is intentional for demonstration purposes.
X = df[["High_School_Average_Mark", "First_Term_Gpa", "Second_Term_Gpa"]]
y = df["GPA_Improvement"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Normalize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/gpa_improvement_scaler.pkl")

# Build neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_absolute_error")

# Train model
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

model.save("model/gpa_improvement_model.h5")

# Predict and evaluate
y_pred = model.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 GPA Improvement NN Model Evaluation")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Plot predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual GPA Improvement")
plt.ylabel("Predicted GPA Improvement")
plt.title("Actual vs Predicted GPA Improvement")
plt.grid(True)
plt.tight_layout()
plt.savefig("model/gpa_improvement_plot.png")
plt.show()
