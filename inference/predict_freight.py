("""inference/predict_freight.py
End-to-end inference script for freight cost prediction.

Features:
- Load a trained model (joblib/pickle)
- Predict for a single input (quantity, dollars), a CSV, or all rows from a SQLite DB
- Write predictions to CSV or print to stdout

This script uses the helper modules in `freight_cost_prediction`.
""")

from pathlib import Path
import sys
import argparse
import pandas as pd
import joblib


# Ensure repo root is on sys.path so freight_cost_prediction can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from freight_cost_prediction import predict as predict_module


def load_model(model_path: str):
	try:
		return joblib.load(model_path)
	except Exception as e:
		raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def predict_single(model, quantity: float, dollars: float):
	return predict_module.predict_freight(model, quantity, dollars)


def predict_from_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
	if not {"Quantity", "Dollars"}.issubset(df.columns):
		raise ValueError("Input dataframe must contain 'Quantity' and 'Dollars' columns")

	X = df[["Quantity", "Dollars"]]
	preds = model.predict(X)
	out = df.copy()
	out["Predicted_Freight"] = preds
	return out


def predict_from_db(model, db_path: str) -> pd.DataFrame:
	try:
		from freight_cost_prediction import data_preprocessing as dp
	except Exception as e:
		raise RuntimeError(f"Failed to import data preprocessing helpers: {e}")

	df = dp.load_vendor_invoice_data(db_path)
	return predict_from_dataframe(model, df)


def parse_args():
	p = argparse.ArgumentParser(description="Freight cost inference script")
	p.add_argument("--model-path", default=str(ROOT / "freight_cost_prediction" / "models" / "predict_freight_model.pkl"), help="Path to trained model file")
	p.add_argument("--quantity", type=float, help="Quantity for single prediction")
	p.add_argument("--dollars", type=float, help="Dollars for single prediction")
	p.add_argument("--input-csv", help="CSV file with Quantity and Dollars columns to predict for")
	p.add_argument("--db-path", help="SQLite DB path; reads vendor_invoice table and predicts for all rows")
	p.add_argument("--output-csv", help="Write predictions to this CSV file (if omitted prints to stdout)")
	return p.parse_args()


def main():
	args = parse_args()

	model = load_model(args.model_path)

	result_df = None

	if args.quantity is not None and args.dollars is not None:
		pred = predict_single(model, args.quantity, args.dollars)
		print(f"Predicted freight: {pred}")
		return

	if args.input_csv:
		df = pd.read_csv(args.input_csv)
		result_df = predict_from_dataframe(model, df)

	if args.db_path:
		result_df = predict_from_db(model, args.db_path)

	if result_df is None:
		print("No input provided. Use --quantity/--dollars, --input-csv, or --db-path.")
		return

	if args.output_csv:
		result_df.to_csv(args.output_csv, index=False)
		print(f"Wrote predictions to {args.output_csv}")
	else:
		print(result_df.head(20).to_string(index=False))


if __name__ == "__main__":
	main()


def predict_freight_cost(input_data, model_path=None):
	"""Wrapper used by app.py: accepts dict, list of dicts, or DataFrame and returns a DataFrame with Predicted_Freight."""
	if model_path is None:
		model_path = str(ROOT / "freight_cost_prediction" / "models" / "predict_freight_model.pkl")

	model = load_model(model_path)

	if isinstance(input_data, dict):
		df = pd.DataFrame([input_data])
	elif isinstance(input_data, list):
		df = pd.DataFrame(input_data)
	elif isinstance(input_data, pd.DataFrame):
		df = input_data.copy()
	else:
		raise ValueError("input_data must be dict, list of dicts, or DataFrame")

	# Normalize possible key names
	if "invoice_dollars" in df.columns and "Dollars" not in df.columns:
		df = df.rename(columns={"invoice_dollars": "Dollars"})
	if "quantity" in df.columns and "Quantity" not in df.columns:
		df = df.rename(columns={"quantity": "Quantity"})

	out = predict_from_dataframe(model, df)
	return out
