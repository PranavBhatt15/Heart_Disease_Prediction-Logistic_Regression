# main.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Step 2: Data Preprocessing
def preprocess_data(df):
    df = df.dropna()
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']
    y = y.map({'Presence': 1, 'Absence': 0})
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Step 3: Model Training with Hyperparameter Tuning
def train_model(X_train, y_train):
    model = LogisticRegression(class_weight='balanced')
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    return best_model

# Step 4: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Step 5: Save Model and Scaler
def save_model_and_scaler(model, scaler, model_file, scaler_file):
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Model saved as {model_file} and scaler saved as {scaler_file}")

# Main Function
if __name__ == "__main__":
    file_path = 'Data/Heart_Disease_Prediction.csv'
    df = load_data(file_path)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = train_model(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)
    save_model_and_scaler(best_model, scaler, 'heart_disease_model.pkl', 'scaler.pkl')
