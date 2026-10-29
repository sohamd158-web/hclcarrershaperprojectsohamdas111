import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed_fault_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "fault_model.pkl"


FEATURES = [
    "voltage_magnitude",
    "current_magnitude",
    "voltage_imbalance",
    "current_imbalance",
    "power",
    "power_variation"
]

TARGET = "fault"


def main():

    print("=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    X = df[FEATURES]
    y = df[TARGET]

    print("\nFault distribution:")
    print(y.value_counts())

    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    

    fault_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    fault_model.fit(X_train, y_train)

   

    predictions = fault_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

   

    print("\nFeature Importance:")

    importance = pd.Series(
        fault_model.feature_importances_,
        index=FEATURES
    ).sort_values(
        ascending=False
    )

    print(importance)

   

    print("\nTraining Isolation Forest...")

    normal_training_data = X_train[
        y_train == "Normal"
    ]

    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    anomaly_model.fit(
        normal_training_data
    )

    
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {
        "fault_model": fault_model,
        "anomaly_model": anomaly_model,
        "features": FEATURES
    }

    joblib.dump(
        model_bundle,
        MODEL_FILE
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(MODEL_FILE)


if __name__ == "__main__":
    main()