# from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
# from model_training import train_models
# from model_evaluation import evaluate_model
# from predict import predict_freight


# def main():

#     db_path = "../inventory.db"

#     # Load data
#     df = load_vendor_invoice_data(db_path)

#     # Prepare features
#     X, y = prepare_features(df)

#     # Split dataset
#     X_train, X_test, y_train, y_test = split_data(X, y)

#     # Train models
#     model1, model2, model3 = train_models(X_train, y_train)

#     # Evaluate models
#     evaluate_model(model1, X_test, y_test, "Linear Regression")
#     evaluate_model(model2, X_test, y_test, "Decision Tree")
#     evaluate_model(model3, X_test, y_test, "Random Forest")

#     # Example prediction
#     freight = predict_freight(model3, quantity=100, dollars=12000)

#     print("\nPredicted Freight Cost:", freight)


# if __name__ == "__main__":
#     main()


from pathlib import Path
import joblib

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_training import train_models
from model_evaluation import evaluate_model
from predict import predict_freight


def main():

    db_path = "../inventory.db"

    # Create models folder
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare features
    X, y = prepare_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model, dt_model, rf_model = train_models(X_train, y_train)

    # Evaluate models
    evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Compare models using R² score
    lr_score = lr_model.score(X_test, y_test)
    dt_score = dt_model.score(X_test, y_test)
    rf_score = rf_model.score(X_test, y_test)

    scores = {
        "Linear Regression": (lr_score, lr_model),
        "Decision Tree": (dt_score, dt_model),
        "Random Forest": (rf_score, rf_model)
    }

    best_model_name = max(scores, key=lambda x: scores[x][0])
    best_model = scores[best_model_name][1]

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")

    # Example prediction
    freight = predict_freight(best_model, quantity=100, dollars=12000)

    print("\nPredicted Freight Cost:", freight)


if __name__ == "__main__":
    main()