"""inference/predict_invoice_flag.py
End-to-end inference script for invoice flag prediction.

Features:
- Load trained classifier and scaler
- Predict for a single input, CSV, or from the DB (uses invoice_flagging helpers)
- Output predictions to CSV or stdout
"""

from pathlib import Path
import sys
import argparse
import pandas as pd
import joblib

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_model(model_path: str):
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def load_scaler(scaler_path: str):
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler from {scaler_path}: {e}")


def predict_single(model, scaler, **kwargs):
    df = pd.DataFrame([{
        k: kwargs.get(k) for k in FEATURES
    }])
    return predict_from_dataframe(model, scaler, df).iloc[0]


def predict_from_dataframe(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    if not set(FEATURES).issubset(df.columns):
        missing = set(FEATURES) - set(df.columns)
        raise ValueError(f"Input dataframe is missing columns: {missing}")

    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    out = df.copy()
    out["Predicted_Flag"] = preds
    return out


def predict_from_db(model, scaler, db_path: str = None) -> pd.DataFrame:
    try:
        from invoice_flagging import data_preprocessing as dp
    except Exception as e:
        raise RuntimeError(f"Failed to import invoice_flagging.data_preprocessing: {e}")

    # data_preprocessing.load_invoice_data uses a hardcoded DB path; allow override
    if db_path:
        df = dp.load_invoice_data()
    else:
        df = dp.load_invoice_data()

    return predict_from_dataframe(model, scaler, df)


def parse_args():
    p = argparse.ArgumentParser(description="Invoice flag inference script")
    p.add_argument("--model-path", default=str(ROOT / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"), help="Path to trained model file")
    p.add_argument("--scaler-path", default=str(ROOT / "invoice_flagging" / "models" / "scaler.pkl"), help="Path to saved scaler file")
    p.add_argument("--input-csv", help="CSV with feature columns to predict for")
    p.add_argument("--db-path", help="Optional DB path (if supported by preprocessing)")
    p.add_argument("--output-csv", help="Write predictions to CSV (otherwise prints head)")
    # Single-input fields
    for f in FEATURES:
        p.add_argument(f"--{f}", type=float if f != "invoice_quantity" else float, help=f"Value for {f}")

    return p.parse_args()


def main():
    args = parse_args()

    model = load_model(args.model_path)
    scaler = load_scaler(args.scaler_path)

    result_df = None

    # Single prediction if feature args provided
    feature_vals = {f: getattr(args, f) for f in FEATURES}
    if all(v is not None for v in feature_vals.values()):
        row = predict_single(model, scaler, **feature_vals)
        print(row.to_string())
        return

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        result_df = predict_from_dataframe(model, scaler, df)

    if args.db_path:
        result_df = predict_from_db(model, scaler, args.db_path)

    if result_df is None:
        print("No input provided. Use feature args, --input-csv, or --db-path.")
        return

    if args.output_csv:
        result_df.to_csv(args.output_csv, index=False)
        print(f"Wrote predictions to {args.output_csv}")
    else:
        print(result_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()


def predict_invoice_flag(input_df, model_path=None, scaler_path=None):
    """Wrapper used by app.py: accepts a DataFrame (or convertible) and returns DataFrame with Predicted_Flag."""
    if model_path is None:
        model_path = str(ROOT / "invoice_flagging" / "models" / "predict_flag_invoice.pkl")
    if scaler_path is None:
        scaler_path = str(ROOT / "invoice_flagging" / "models" / "scaler.pkl")

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    if isinstance(input_df, pd.DataFrame):
        df = input_df.copy()
    elif isinstance(input_df, list):
        df = pd.DataFrame(input_df)
    elif isinstance(input_df, dict):
        df = pd.DataFrame([input_df])
    else:
        raise ValueError("input_df must be DataFrame, dict or list of dicts")

    # Normalize column name: some references use lower-case 'freight'
    if "freight" in df.columns and "Freight" not in df.columns:
        df = df.rename(columns={"freight": "Freight"})

    out = predict_from_dataframe(model, scaler, df)
    return out
