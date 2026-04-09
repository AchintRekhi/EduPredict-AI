# train_model.py
# Neural Network Model Training Script to Predict First Year Persistence
# This script trains a neural network model to predict whether a student
# will persist in their first year of studies.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessor():
    """Build a ColumnTransformer that scales numeric features, passes through
    ordinal features, and one-hot encodes categorical features."""
    num_features = ['First_Term_Gpa', 'Second_Term_Gpa', 'High_School_Average_Mark', 'Math_Score']
    ordinal_features = ['English_Grade', 'Age_Group', 'Previous_Education']
    cat_features = ['First_Language', 'Funding', 'Gender', 'FastTrack', 'Coop', 'Residency']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('ord', 'passthrough', ordinal_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ])
    return preprocessor


def build_model(input_dim):
    """Build a 3-hidden-layer feedforward NN for binary classification."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Load Data
df = pd.read_csv("data/Student_Data.csv")
X = df.drop(columns=["FirstYearPersistence"])
y = df["FirstYearPersistence"]

# Preprocess Data
preprocessor = build_preprocessor()
X_processed = preprocessor.fit_transform(X)
joblib.dump(preprocessor, "model/preprocessor.pkl")

# Split Data and Handle Class Imbalance with SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, stratify=y, random_state=42)
X_train_sm, y_train_sm = SMOTE().fit_resample(X_train, y_train)

# Calculate Class Weights for Balanced Training
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_sm),
    y=y_train_sm
)
class_weight_dict = dict(zip(np.unique(y_train_sm), class_weights))

# Build and Train Model
model = build_model(X_train_sm.shape[1])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train_sm, y_train_sm,
          validation_split=0.2,
          epochs=100,
          batch_size=32,
          class_weight=class_weight_dict,
          callbacks=[early_stop],
          verbose=1)

# Save the Trained Model
model.save("model/trained_model.h5")

# Evaluate the Model
y_probs = model.predict(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_thresh = thresholds[np.argmax(f1_scores)]

print("Best threshold:", best_thresh)

y_pred = (y_probs > best_thresh).astype("int32")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
