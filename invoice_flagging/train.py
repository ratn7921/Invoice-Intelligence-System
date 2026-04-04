from pathlib import Path
import joblib

from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from modeling_evaluation import train_random_forest, evaluate_classifier


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    # Create models folder
    Path("models").mkdir(exist_ok=True)

    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # Scale
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, "models/scaler.pkl"
    )

    # Train
    grid_search = train_random_forest(X_train_scaled, y_train)

    # Evaluate
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save best model
    joblib.dump(
        grid_search.best_estimator_,
        "models/predict_flag_invoice.pkl"
    )

    print("\n✅ Model saved successfully!")


if __name__ == "__main__":
    main()