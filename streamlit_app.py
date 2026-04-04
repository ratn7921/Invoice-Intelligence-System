"""Streamlit UI for Invoice Intelligence System

Provides two tools:
- Freight Cost Prediction
- Invoice Flagging

Usage: `streamlit run streamlit_app.py`
"""

from pathlib import Path
import sys
import io
import joblib
import pandas as pd
import streamlit as st

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Invoice Intelligence", layout="wide")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource
def load_scaler(path: str):
    return joblib.load(path)


def freight_tab():
    st.header("Freight Cost Prediction")

    default_model = ROOT / "freight_cost_prediction" / "models" / "predict_freight_model.pkl"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Single prediction")
        qty = st.number_input("Quantity", min_value=0.0, value=1.0)
        dollars = st.number_input("Dollars", min_value=0.0, value=100.0)
        model_path = st.text_input("Model path", value=str(default_model))
        if st.button("Predict freight"):
            try:
                model = load_model(model_path)
                inp = pd.DataFrame({"Quantity": [qty], "Dollars": [dollars]})
                pred = model.predict(inp)[0]
                st.success(f"Predicted freight: {pred:.4f}")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("Batch prediction (CSV)")
        uploaded = st.file_uploader("Upload CSV with Quantity and Dollars columns", type=["csv"] , key="freight_csv")
        out_path = st.text_input("Output CSV path (optional)")
        model_path2 = st.text_input("Model path (batch)", value=str(default_model), key="model_path_batch")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if not {"Quantity", "Dollars"}.issubset(df.columns):
                    st.error("CSV must contain 'Quantity' and 'Dollars' columns")
                else:
                    model = load_model(model_path2)
                    preds = model.predict(df[["Quantity", "Dollars"]])
                    df_out = df.copy()
                    df_out["Predicted_Freight"] = preds
                    st.dataframe(df_out)
                    csv = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions", csv, file_name="freight_predictions.csv", mime="text/csv")
                    if out_path:
                        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                        df_out.to_csv(out_path, index=False)
                        st.success(f"Wrote to {out_path}")
            except Exception as e:
                st.error(f"Error reading CSV or predicting: {e}")


def invoice_tab():
    st.header("Invoice Flagging")

    default_model = ROOT / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
    default_scaler = ROOT / "invoice_flagging" / "models" / "scaler.pkl"

    FEATURES = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars",
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Single prediction")
        inputs = {}
        for f in FEATURES:
            inputs[f] = st.number_input(f, value=0.0)
        model_path = st.text_input("Model path", value=str(default_model), key="inv_model_single")
        scaler_path = st.text_input("Scaler path", value=str(default_scaler), key="inv_scaler_single")
        if st.button("Predict flag"):
            try:
                model = load_model(model_path)
                scaler = load_scaler(scaler_path)
                df = pd.DataFrame([{k: v for k, v in inputs.items()}])
                X = df[FEATURES]
                Xs = scaler.transform(X)
                pred = model.predict(Xs)[0]
                st.success(f"Predicted flag (1 = suspicious): {int(pred)}")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("Batch prediction (CSV)")
        uploaded = st.file_uploader("Upload CSV with invoice feature columns", type=["csv"], key="inv_csv")
        out_path = st.text_input("Output CSV path (optional)", key="inv_out_path")
        model_path2 = st.text_input("Model path (batch)", value=str(default_model), key="inv_model_batch")
        scaler_path2 = st.text_input("Scaler path (batch)", value=str(default_scaler), key="inv_scaler_batch")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                missing = set(FEATURES) - set(df.columns)
                if missing:
                    st.error(f"CSV missing columns: {missing}")
                else:
                    model = load_model(model_path2)
                    scaler = load_scaler(scaler_path2)
                    X = df[FEATURES]
                    Xs = scaler.transform(X)
                    preds = model.predict(Xs)
                    df_out = df.copy()
                    df_out["Predicted_Flag"] = preds
                    st.dataframe(df_out)
                    csv = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions", csv, file_name="invoice_predictions.csv", mime="text/csv")
                    if out_path:
                        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                        df_out.to_csv(out_path, index=False)
                        st.success(f"Wrote to {out_path}")
            except Exception as e:
                st.error(f"Error reading CSV or predicting: {e}")


def main():
    st.title("Invoice Intelligence System")
    st.write("Predict freight costs and flag suspicious invoices")

    task = st.sidebar.selectbox("Choose tool", ["Freight Prediction", "Invoice Flagging"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Models are loaded from the project's models folders by default.")

    if task == "Freight Prediction":
        freight_tab()
    else:
        invoice_tab()


if __name__ == "__main__":
    main()
