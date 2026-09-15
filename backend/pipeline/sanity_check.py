import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def main():
    print("=" * 70)
    print("ARTIFACT LOAD SANITY CHECK")
    print("=" * 70)

    # 1. Load full feature order
    feature_order_path = os.path.join(MODELS_DIR, "full_feature_order.json")
    with open(feature_order_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f)
    print(f"[1/4] Loaded full_feature_order.json: {len(feature_order)} features expected.")

    # 2. Load StandardScaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[2/4] Loaded scaler.pkl: expects {len(scaler.mean_)} features.")

    # 3. Load Feature Selector
    selector_path = os.path.join(MODELS_DIR, "feature_selector.pkl")
    with open(selector_path, "rb") as f:
        selector = pickle.load(f)
    n_selected = int(selector.get_support().sum())
    print(f"[3/4] Loaded feature_selector.pkl: selects {n_selected} features.")

    # 4. Load Trained Proposed ANN
    model_path = os.path.join(MODELS_DIR, "phishing_ann_selected30.keras")
    model = tf.keras.models.load_model(model_path)
    print(f"[4/4] Loaded phishing_ann_selected30.keras: input shape = {model.input_shape}.")

    # 5. Load one random sample from the original CSV
    csv_path = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")
    print(f"\nSampling 1 row from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    # Take a random sample with fixed seed for deterministic reporting
    sample_row = df.sample(n=1, random_state=42)
    sample_idx = sample_row.index[0]
    raw_label = sample_row["label"].values[0]
    sample_url = sample_row["URL"].values[0] if "URL" in sample_row.columns else "N/A"

    print(f"Sample Index: {sample_idx}")
    print(f"Sample URL  : {sample_url}")
    print(f"True Label  : {raw_label} ({'Legitimate' if raw_label == 1 else 'Phishing'})")

    # Align sample features with full_feature_order
    X_sample = sample_row[feature_order]

    # Pipeline: scaler.transform -> feature_selector.transform -> model.predict
    X_scaled = scaler.transform(X_sample)
    X_selected = selector.transform(X_scaled)
    prob_legit = float(model.predict(X_selected, verbose=0)[0][0])
    prob_phish = 1.0 - prob_legit
    pred_label = 1 if prob_legit >= 0.5 else 0

    print("\n" + "-" * 70)
    print("PIPELINE INFERENCE RESULTS:")
    print("-" * 70)
    print(f"Raw ANN Output (P_legitimate): {prob_legit:.6f}")
    print(f"Phishing Probability (P_phish): {prob_phish:.6f} ({prob_phish * 100:.2f}%)")
    print(f"Predicted Class               : {pred_label} ({'Legitimate' if pred_label == 1 else 'Phishing'})")
    print(f"Ground Truth Match            : {'MATCH (CORRECT)' if pred_label == raw_label else 'MISMATCH'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
