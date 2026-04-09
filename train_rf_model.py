# train_rf_model.py
# # Random Forest Model Training Script to Predict First Year Persistence
# This script trains a Random Forest model to predict whether a student will persist in their first year.
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/Student_Data.csv")

# Define target and features
target = 'FirstYearPersistence'
X = df.drop(columns=[target])
y = df[target]

# Define column types
num_features = ['First_Term_Gpa', 'Second_Term_Gpa', 'High_School_Average_Mark', 'Math_Score']
ordinal_features = ['English_Grade', 'Age_Group', 'Previous_Education']
cat_features = ['First_Language', 'Funding', 'Gender', 'FastTrack', 'Coop', 'Residency']

# Column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('ord', 'passthrough', ordinal_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])

# Transform features
X_processed = preprocessor.fit_transform(X)

# Save the preprocessor
joblib.dump(preprocessor, "model/rf_preprocessor.pkl")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.25, stratify=y, random_state=42
)

# Handle class imbalance
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), weights))

# Train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weight_dict,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "model/rf_model.pkl")

# Predict probabilities
y_probs = rf_model.predict_proba(X_test)[:, 1]

# Precision-recall curve and threshold tuning
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_thresh = thresholds[np.argmax(f1_scores)]
pr_auc = auc(recall, precision)

# Final predictions using best threshold
y_pred = (y_probs > best_thresh).astype("int32")

# Evaluation
print("✅ Training Complete")
print("📉 Best Threshold (F1-score):", round(best_thresh, 4))
print("🔍 PR-AUC:", round(pr_auc, 4))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
