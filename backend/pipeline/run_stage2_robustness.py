"""
Stage 2: Robustness Validation Pipeline
========================================================================================
Research Paper: "URL-Based Phishing Website Detection Using Machine Learning and Artificial Neural Networks"
Authors: Christy Francis, Saksham Joshi, Aadesh Patil
Department of Computer Engineering, Pimpri Chinchwad College of Engineering, Pune

This script executes three independent, rigorous robustness checks:
- Part A: Stratified 5-Fold Cross-Validation on PhiUSIIL (Zero-Leakage per-fold MI & Scaling)
- Part B: Cross-Dataset Generalization Test on UCI Phishing Websites Dataset (Schema Alignment & External Test)
- Part C: URL-Only Feature Scope Ablation (20 Pre-Crawl Lexical Features vs Full Features)
- Final Consolidation: Comprehensive Comparison Summary Table & Discussion Synthesis
"""

import os
import sys
import time
import pickle
import json
import warnings
import numpy as np
import pandas as pd

# Suppress non-critical warnings & TensorFlow info logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Scikit-Learn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# XGBoost
from xgboost import XGBClassifier

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def compute_mi(X, y):
    """Score function for SelectKBest using mutual_info_classif with fixed seed."""
    return mutual_info_classif(X, y, random_state=SEED)


def build_ann(input_dim):
    """Build the proposed ANN architecture matching Stage 1 exactly."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu", name="Dense_1"),
        Dropout(0.30, name="Dropout_1"),
        Dense(64, activation="relu", name="Dense_2"),
        Dropout(0.30, name="Dropout_2"),
        Dense(32, activation="relu", name="Dense_3"),
        Dense(1, activation="sigmoid", name="Output_Sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def compute_metrics(y_true_raw, y_prob_phish):
    """
    Compute binary classification metrics treating Phishing as the positive class.
    In the raw dataset:
      label = 1 -> Legitimate
      label = 0 -> Phishing
    For evaluation:
      Actual Phishing: y_true_phish = 1 (when raw == 0)
      Predicted Phishing: y_pred_phish = 1 (when y_prob_phish >= 0.5)
    """
    y_true_phish = (y_true_raw == 0).astype(int)
    y_pred_phish = (y_prob_phish >= 0.5).astype(int)

    acc = accuracy_score(y_true_phish, y_pred_phish)
    prec = precision_score(y_true_phish, y_pred_phish, zero_division=0)
    rec = recall_score(y_true_phish, y_pred_phish, zero_division=0)
    f1 = f1_score(y_true_phish, y_pred_phish, zero_division=0)
    try:
        auc = roc_auc_score(y_true_phish, y_prob_phish)
    except Exception:
        auc = float("nan")

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc
    }


def run_part_c(df_phi_clean, base_dir, results_dir, models_dir):
    """
    PART C — URL-ONLY FEATURE SCOPE ABLATION (NO WEBPAGE-CRAWL FEATURES)
    Trains ANN on 20 pre-crawl lexical/host features and persists all artifacts:
    - models/phishing_ann_urlonly20.keras
    - models/urlonly_scaler.pkl
    - models/urlonly_feature_selector.pkl
    - models/urlonly_selected_features.json
    """
    print("\n" + "=" * 85)
    print("PART C: URL-ONLY FEATURE SCOPE ABLATION (NO WEBPAGE CRAWL FEATURES)")
    print("=" * 85)
    print("Testing performance using strictly the 20 pre-crawl lexical/host features.")

    url_only_features = [
        "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate", "URLCharProb",
        "TLDLength", "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
        "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
        "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
        "SpacialCharRatioInURL", "IsHTTPS"
    ]
    print(f"Selected {len(url_only_features)} URL-only features:")
    print(url_only_features)

    # Ensure all exist in dataset
    missing_url_cols = [c for c in url_only_features if c not in df_phi_clean.columns]
    if missing_url_cols:
        raise ValueError(f"Missing required URL-only columns: {missing_url_cols}")

    X_url = df_phi_clean[url_only_features].values
    y_url = df_phi_clean["label"].values

    X_tr_u, X_te_u, y_tr_u, y_te_u = train_test_split(
        X_url,
        y_url,
        test_size=0.20,
        random_state=SEED,
        stratify=y_url
    )

    # 1. Feature Scaling
    scaler_u = StandardScaler()
    X_tr_u_scaled = scaler_u.fit_transform(X_tr_u)
    X_te_u_scaled = scaler_u.transform(X_te_u)

    # 2. Feature Selector (SelectKBest with k=20)
    selector_u = SelectKBest(score_func=mutual_info_classif, k=len(url_only_features))
    X_tr_u_sel = selector_u.fit_transform(X_tr_u_scaled, y_tr_u)
    X_te_u_sel = selector_u.transform(X_te_u_scaled)

    # 3. Train proposed ANN architecture (input dim 20)
    ann_url = build_ann(input_dim=len(url_only_features))
    es_u = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    ann_url.fit(
        X_tr_u_sel,
        y_tr_u,
        validation_split=0.20,
        epochs=40,
        batch_size=512,
        callbacks=[es_u],
        verbose=0
    )

    prob_url_legit = ann_url.predict(X_te_u_sel, batch_size=1024, verbose=0).ravel()
    prob_url_phish = 1.0 - prob_url_legit
    metrics_url_only = compute_metrics(y_te_u, prob_url_phish)

    # 4. Save Standalone Part C Artifacts
    os.makedirs(models_dir, exist_ok=True)

    # Trained ANN as models/phishing_ann_urlonly20.keras
    ann_url_path = os.path.join(models_dir, "phishing_ann_urlonly20.keras")
    ann_url.save(ann_url_path)
    print(f"\n[Artifact Saved] Trained URL-Only ANN: {ann_url_path} ({os.path.getsize(ann_url_path):,} bytes)")

    # StandardScaler as models/urlonly_scaler.pkl
    scaler_u_path = os.path.join(models_dir, "urlonly_scaler.pkl")
    with open(scaler_u_path, "wb") as f:
        pickle.dump(scaler_u, f)
    print(f"[Artifact Saved] URL-Only Scaler: {scaler_u_path} ({os.path.getsize(scaler_u_path):,} bytes)")

    # SelectKBest selector as models/urlonly_feature_selector.pkl
    selector_u_path = os.path.join(models_dir, "urlonly_feature_selector.pkl")
    with open(selector_u_path, "wb") as f:
        pickle.dump(selector_u, f)
    print(f"[Artifact Saved] URL-Only Feature Selector: {selector_u_path} ({os.path.getsize(selector_u_path):,} bytes)")

    # Exact list of 20 feature names as models/urlonly_selected_features.json
    selected_20_names = [url_only_features[i] for i in range(len(url_only_features)) if selector_u.get_support()[i]]
    urlonly_features_path = os.path.join(models_dir, "urlonly_selected_features.json")
    with open(urlonly_features_path, "w", encoding="utf-8") as f:
        json.dump(selected_20_names, f, indent=2)
    print(f"[Artifact Saved] URL-Only Selected Features JSON: {urlonly_features_path} ({os.path.getsize(urlonly_features_path):,} bytes)")

    # Load Stage 1 baseline for comparison
    stage1_metrics = {
        "Accuracy": 0.99966,
        "Precision": 0.99975,
        "Recall": 0.99945,
        "F1": 0.99960,
        "ROC-AUC": 0.999996
    }

    print("\n--- Part C: URL-Only Ablation Results ---")
    print(f"Stage 1 Proposed ANN (30 MI Features, Full Content):")
    print(f"  Accuracy:  {stage1_metrics['Accuracy']*100:.3f}%")
    print(f"  F1-Score:  {stage1_metrics['F1']*100:.3f}%")
    print(f"  ROC-AUC:   {stage1_metrics['ROC-AUC']:.6f}")

    print(f"\nPart C URL-Only ANN (20 Lexical Features, Pre-Crawl):")
    print(f"  Accuracy:  {metrics_url_only['Accuracy']*100:.3f}%")
    print(f"  Precision: {metrics_url_only['Precision']*100:.3f}%")
    print(f"  Recall:    {metrics_url_only['Recall']*100:.3f}%")
    print(f"  F1-Score:  {metrics_url_only['F1']*100:.3f}%")
    print(f"  ROC-AUC:   {metrics_url_only['ROC-AUC']:.6f}")

    df_url_ablation = pd.DataFrame([
        {"Model_Scope": "Stage 1 Full Content ANN (30 MI Features)", **stage1_metrics},
        {"Model_Scope": "Part C URL-Only ANN (20 Pre-Crawl Lexical Features)", **metrics_url_only}
    ])
    url_csv_path = os.path.join(results_dir, "url_only_ablation_results.csv")
    df_url_ablation.to_csv(url_csv_path, index=False)
    print(f"Saved URL-Only Ablation results to: {url_csv_path}")

    return metrics_url_only


def main():
    print("=" * 85)
    print("STAGE 2: ROBUSTNESS VALIDATION (Cross-Dataset, K-Fold, URL-Only Scope)")
    print("=" * 85)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(project_root, "results")
    models_dir = os.path.join(project_root, "models")
    data_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    phi_path = os.path.join(data_dir, "PhiUSIIL_Phishing_URL_Dataset.csv")
    uci_path = os.path.join(data_dir, "UCI_Phishing_Websites.csv")

    if not os.path.exists(phi_path):
        raise FileNotFoundError(f"PhiUSIIL dataset not found at {phi_path}")
    if not os.path.exists(uci_path):
        raise FileNotFoundError(f"UCI dataset not found at {uci_path}")

    # Load and clean PhiUSIIL dataset
    print("\n[DATA PREPARATION] Loading PhiUSIIL dataset...")
    df_phi = pd.read_csv(phi_path)
    df_phi.dropna(inplace=True)
    df_phi.drop_duplicates(inplace=True)

    # Drop non-predictive identifiers and leaking columns
    id_cols = ["FILENAME", "URL", "Domain", "Title", "TLD"]
    leaking_cols = ["URLSimilarityIndex", "TLDLegitimateProb"]
    cols_to_drop = [c for c in (id_cols + leaking_cols) if c in df_phi.columns]

    df_phi_clean = df_phi.drop(columns=cols_to_drop)
    X_full = df_phi_clean.drop("label", axis=1)
    y_full = df_phi_clean["label"].values
    feature_names = list(X_full.columns)
    print(f"PhiUSIIL usable dataset shape: {X_full.shape} features, {len(y_full)} samples.")
    print(f"Target distribution: Legitimate (1) = {np.sum(y_full == 1):,}, Phishing (0) = {np.sum(y_full == 0):,}")

    if "--part-c-only" in sys.argv:
        print("\nExecuting Part C (URL-Only Feature Scope Ablation) standalone...")
        run_part_c(df_phi_clean, base_dir, results_dir, models_dir)
        print("\nPart C standalone execution complete.")
        return

    # =========================================================================
    # PART A — STRATIFIED 5-FOLD CROSS-VALIDATION ON PHIUSIIL
    # =========================================================================
    print("\n" + "=" * 85)
    print("PART A: STRATIFIED 5-FOLD CROSS-VALIDATION ON PHIUSIIL")
    print("=" * 85)
    print("Verifying that results are not an artifact of a single train/test split.")
    print("CRITICAL: StandardScaler and SelectKBest(k=30) are fitted exclusively")
    print("on each fold's training portion to guarantee zero data leakage.\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_metrics_ann = []
    fold_metrics_rf = []
    fold_metrics_xgb = []

    X_np = X_full.values

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_np, y_full), 1):
        print(f"--- FOLD {fold}/5 ---")
        X_train_raw, X_test_raw = X_np[train_idx], X_np[test_idx]
        y_train_fold, y_test_fold = y_full[train_idx], y_full[test_idx]

        # 1. Feature Scaling (fresh per fold)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 2. Feature Selection: Top 30 MI Features (fresh per fold)
        t0_sel = time.time()
        selector = SelectKBest(score_func=compute_mi, k=30)
        X_train_sel30 = selector.fit_transform(X_train_scaled, y_train_fold)
        X_test_sel30 = selector.transform(X_test_scaled)
        t_sel = time.time() - t0_sel
        print(f"  Fold {fold}: MI Feature Selection completed in {t_sel:.2f}s")

        # 3. Train Proposed ANN
        ann = build_ann(input_dim=30)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        ann.fit(
            X_train_sel30,
            y_train_fold,
            validation_split=0.20,
            epochs=40,
            batch_size=512,
            callbacks=[es],
            verbose=0
        )
        prob_ann_legit = ann.predict(X_test_sel30, batch_size=1024, verbose=0).ravel()
        prob_ann_phish = 1.0 - prob_ann_legit
        metrics_ann = compute_metrics(y_test_fold, prob_ann_phish)
        fold_metrics_ann.append(metrics_ann)
        print(f"  Fold {fold} Proposed ANN: Acc={metrics_ann['Accuracy']*100:.3f}%, F1={metrics_ann['F1']*100:.3f}%, ROC-AUC={metrics_ann['ROC-AUC']:.5f}")

        # 4. Train Random Forest Baseline
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        rf.fit(X_train_sel30, y_train_fold)
        # In predict_proba: class 0 is Phishing (column 0), class 1 is Legitimate (column 1)
        prob_rf_phish = rf.predict_proba(X_test_sel30)[:, 0]
        metrics_rf = compute_metrics(y_test_fold, prob_rf_phish)
        fold_metrics_rf.append(metrics_rf)
        print(f"  Fold {fold} Random Forest: Acc={metrics_rf['Accuracy']*100:.3f}%, F1={metrics_rf['F1']*100:.3f}%, ROC-AUC={metrics_rf['ROC-AUC']:.5f}")

        # 5. Train XGBoost Baseline
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED, eval_metric='logloss', n_jobs=-1)
        xgb.fit(X_train_sel30, y_train_fold)
        prob_xgb_phish = xgb.predict_proba(X_test_sel30)[:, 0]
        metrics_xgb = compute_metrics(y_test_fold, prob_xgb_phish)
        fold_metrics_xgb.append(metrics_xgb)
        print(f"  Fold {fold} XGBoost:       Acc={metrics_xgb['Accuracy']*100:.3f}%, F1={metrics_xgb['F1']*100:.3f}%, ROC-AUC={metrics_xgb['ROC-AUC']:.5f}\n")

    # Build K-Fold Results Table
    metric_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    kfold_rows = []

    models_dict = {
        "Proposed ANN (30 MI)": fold_metrics_ann,
        "Random Forest (30 MI)": fold_metrics_rf,
        "XGBoost (30 MI)": fold_metrics_xgb
    }

    for model_name, folds in models_dict.items():
        for m in metric_keys:
            vals = [f[m] for f in folds]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            row = {
                "Model": model_name,
                "Metric": m,
                "Fold 1": vals[0],
                "Fold 2": vals[1],
                "Fold 3": vals[2],
                "Fold 4": vals[3],
                "Fold 5": vals[4],
                "Mean": mean_val,
                "Std Dev": std_val
            }
            kfold_rows.append(row)

    df_kfold = pd.DataFrame(kfold_rows)
    kfold_csv_path = os.path.join(results_dir, "kfold_cross_validation_results.csv")
    df_kfold.to_csv(kfold_csv_path, index=False)
    print(f"Saved 5-Fold Cross-Validation results to: {kfold_csv_path}")

    print("\n--- Part A: 5-Fold Cross-Validation Summary (Mean ± Std Dev) ---")
    summary_kfold = []
    for model_name in models_dict:
        sub = df_kfold[df_kfold["Model"] == model_name]
        acc_row = sub[sub["Metric"] == "Accuracy"].iloc[0]
        f1_row = sub[sub["Metric"] == "F1"].iloc[0]
        auc_row = sub[sub["Metric"] == "ROC-AUC"].iloc[0]
        print(f"{model_name:25}: Acc = {acc_row['Mean']*100:.3f}% ± {acc_row['Std Dev']*100:.4f}% | F1 = {f1_row['Mean']*100:.3f}% ± {f1_row['Std Dev']*100:.4f}% | AUC = {auc_row['Mean']:.6f} ± {auc_row['Std Dev']:.6f}")

    # =========================================================================
    # PART B — CROSS-DATASET GENERALIZATION TEST (UCI PHISHING WEBSITES)
    # =========================================================================
    print("\n" + "=" * 85)
    print("PART B: CROSS-DATASET GENERALIZATION TEST (UCI PHISHING WEBSITES)")
    print("=" * 85)
    print("Evaluating external generalizability against Mohammad et al. (11,055 rows).")

    df_uci = pd.read_csv(uci_path)
    print(f"Loaded UCI dataset: {df_uci.shape[0]} rows, {df_uci.shape[1]} columns.")
    # UCI label: Result in {-1, 1}. Map -1 -> 0 (Phishing), 1 -> 1 (Legitimate)
    y_uci_raw = np.where(df_uci["Result"] == 1, 1, 0)
    print(f"UCI class distribution: Legitimate (1) = {np.sum(y_uci_raw == 1):,}, Phishing (0) = {np.sum(y_uci_raw == 0):,}")

    # Explicit Conceptually Equivalent Feature Alignment
    # Documenting the exact mapping schema:
    mapping_docs = [
        {
            "PhiUSIIL_Feature": "IsDomainIP",
            "UCI_Feature": "having_IP_Address",
            "PhiUSIIL_Original_Definition": "Binary flag (1 = IP address in domain, 0 = No)",
            "UCI_Original_Definition": "Categorical (1 = Legitimate/No IP, -1 = Phishing/IP)",
            "Harmonized_Encoding": "1 = Legitimate (No IP), -1 = Phishing (Has IP)"
        },
        {
            "PhiUSIIL_Feature": "URLLength",
            "UCI_Feature": "URL_Length",
            "PhiUSIIL_Original_Definition": "Continuous character count of URL string",
            "UCI_Original_Definition": "Rule-based (1 = <54 chars, 0 = 54-75 chars, -1 = >75 chars)",
            "Harmonized_Encoding": "Discretized via Mohammad et al. thresholds: <54 -> 1, 54-75 -> 0, >75 -> -1"
        },
        {
            "PhiUSIIL_Feature": "NoOfSubDomain",
            "UCI_Feature": "having_Sub_Domain",
            "PhiUSIIL_Original_Definition": "Integer count of subdomain levels",
            "UCI_Original_Definition": "Rule-based (1 = No subdomain, 0 = 1 subdomain, -1 = Multi-subdomain)",
            "Harmonized_Encoding": "Discretized: 0 -> 1, 1 -> 0, >=2 -> -1"
        },
        {
            "PhiUSIIL_Feature": "IsHTTPS",
            "UCI_Feature": "SSLfinal_State",
            "PhiUSIIL_Original_Definition": "Binary flag (1 = HTTPS, 0 = HTTP)",
            "UCI_Original_Definition": "Categorical (1 = Trusted SSL, 0 = Suspicious, -1 = No SSL/Untrusted)",
            "Harmonized_Encoding": "PhiUSIIL: 1 -> 1, 0 -> -1; UCI preserves original {-1, 0, 1}"
        },
        {
            "PhiUSIIL_Feature": "HasFavicon",
            "UCI_Feature": "Favicon",
            "PhiUSIIL_Original_Definition": "Binary flag (1 = Favicon present, 0 = Absent)",
            "UCI_Original_Definition": "Categorical (1 = Same domain, -1 = External/Absent)",
            "Harmonized_Encoding": "PhiUSIIL: 1 -> 1, 0 -> -1; UCI: {-1, 1}"
        },
        {
            "PhiUSIIL_Feature": "NoOfiFrame",
            "UCI_Feature": "Iframe",
            "PhiUSIIL_Original_Definition": "Integer count of <iframe> tags in DOM",
            "UCI_Original_Definition": "Categorical (1 = No invisible iframe, -1 = Invisible iframe)",
            "Harmonized_Encoding": "PhiUSIIL: 0 -> 1, >0 -> -1; UCI: {-1, 1}"
        },
        {
            "PhiUSIIL_Feature": "NoOfPopup",
            "UCI_Feature": "popUpWidnow",
            "PhiUSIIL_Original_Definition": "Integer count of JavaScript popup dialogue boxes",
            "UCI_Original_Definition": "Categorical (1 = Normal popup/none, -1 = Popup with fields)",
            "Harmonized_Encoding": "PhiUSIIL: 0 -> 1, >0 -> -1; UCI: {-1, 1}"
        },
        {
            "PhiUSIIL_Feature": "NoOfExternalRef / TotalRef",
            "UCI_Feature": "Request_URL",
            "PhiUSIIL_Original_Definition": "Ratio of external asset references to total document references",
            "UCI_Original_Definition": "Categorical (1 = <22% ext objects, 0 = 22-61% ext, -1 = >61% ext)",
            "Harmonized_Encoding": "Discretized via Mohammad et al. thresholds: <22% -> 1, 22-61% -> 0, >61% -> -1"
        }
    ]

    df_mapping = pd.DataFrame(mapping_docs)
    mapping_csv_path = os.path.join(results_dir, "uci_feature_mapping.csv")
    df_mapping.to_csv(mapping_csv_path, index=False)
    print(f"Documented {len(df_mapping)} conceptually equivalent features. Saved mapping to: {mapping_csv_path}")

    # Construct harmonized feature matrices
    # 1. IP Address
    phi_f1 = np.where(df_phi["IsDomainIP"] == 1, -1, 1)
    uci_f1 = df_uci["having_IP_Address"].values

    # 2. URL Length
    phi_f2 = np.where(df_phi["URLLength"] < 54, 1, np.where(df_phi["URLLength"] <= 75, 0, -1))
    uci_f2 = df_uci["URL_Length"].values

    # 3. Subdomains
    phi_f3 = np.where(df_phi["NoOfSubDomain"] == 0, 1, np.where(df_phi["NoOfSubDomain"] == 1, 0, -1))
    uci_f3 = df_uci["having_Sub_Domain"].values

    # 4. HTTPS / SSL
    phi_f4 = np.where(df_phi["IsHTTPS"] == 1, 1, -1)
    uci_f4 = df_uci["SSLfinal_State"].values

    # 5. Favicon
    phi_f5 = np.where(df_phi["HasFavicon"] == 1, 1, -1)
    uci_f5 = df_uci["Favicon"].values

    # 6. IFrame
    phi_f6 = np.where(df_phi["NoOfiFrame"] == 0, 1, -1)
    uci_f6 = df_uci["Iframe"].values

    # 7. Popup Window
    phi_f7 = np.where(df_phi["NoOfPopup"] == 0, 1, -1)
    uci_f7 = df_uci["popUpWidnow"].values

    # 8. External References / Request URL
    tot_ref = df_phi["NoOfSelfRef"] + df_phi["NoOfEmptyRef"] + df_phi["NoOfExternalRef"]
    ext_ratio = df_phi["NoOfExternalRef"] / np.maximum(tot_ref, 1)
    phi_f8 = np.where(ext_ratio < 0.22, 1, np.where(ext_ratio <= 0.61, 0, -1))
    uci_f8 = df_uci["Request_URL"].values

    X_phi_common = np.column_stack([phi_f1, phi_f2, phi_f3, phi_f4, phi_f5, phi_f6, phi_f7, phi_f8])
    y_phi_common = df_phi["label"].values

    X_uci_common = np.column_stack([uci_f1, uci_f2, uci_f3, uci_f4, uci_f5, uci_f6, uci_f7, uci_f8])

    # Train-test split on PhiUSIIL (80/20 stratified)
    X_tr_com, X_te_com, y_tr_com, y_te_com = train_test_split(
        X_phi_common,
        y_phi_common,
        test_size=0.20,
        random_state=SEED,
        stratify=y_phi_common
    )

    # Scale using scaler fitted strictly on PhiUSIIL training split
    scaler_common = StandardScaler()
    X_tr_com_scaled = scaler_common.fit_transform(X_tr_com)
    X_te_com_scaled = scaler_common.transform(X_te_com)
    X_uci_scaled = scaler_common.transform(X_uci_common)

    # Retrain ANN with identical architecture on the 8 common features
    ann_common = build_ann(input_dim=8)
    es_com = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    ann_common.fit(
        X_tr_com_scaled,
        y_tr_com,
        validation_split=0.20,
        epochs=40,
        batch_size=512,
        callbacks=[es_com],
        verbose=0
    )

    # In-Dataset Evaluation: PhiUSIIL Test Split (8 common features)
    prob_phi_com_legit = ann_common.predict(X_te_com_scaled, batch_size=1024, verbose=0).ravel()
    prob_phi_com_phish = 1.0 - prob_phi_com_legit
    metrics_phi_common = compute_metrics(y_te_com, prob_phi_com_phish)

    # External Validation: Full UCI Dataset
    prob_uci_legit = ann_common.predict(X_uci_scaled, batch_size=1024, verbose=0).ravel()
    prob_uci_phish = 1.0 - prob_uci_legit
    metrics_uci_external = compute_metrics(y_uci_raw, prob_uci_phish)

    print("\n--- Part B: Cross-Dataset Generalization Results ---")
    print(f"PhiUSIIL Test Split (8 Common Features):")
    print(f"  Accuracy:  {metrics_phi_common['Accuracy']*100:.3f}%")
    print(f"  Precision: {metrics_phi_common['Precision']*100:.3f}%")
    print(f"  Recall:    {metrics_phi_common['Recall']*100:.3f}%")
    print(f"  F1-Score:  {metrics_phi_common['F1']*100:.3f}%")
    print(f"  ROC-AUC:   {metrics_phi_common['ROC-AUC']:.5f}")

    print(f"\nUCI External Dataset (Held-Out Generalization):")
    print(f"  Accuracy:  {metrics_uci_external['Accuracy']*100:.3f}%")
    print(f"  Precision: {metrics_uci_external['Precision']*100:.3f}%")
    print(f"  Recall:    {metrics_uci_external['Recall']*100:.3f}%")
    print(f"  F1-Score:  {metrics_uci_external['F1']*100:.3f}%")
    print(f"  ROC-AUC:   {metrics_uci_external['ROC-AUC']:.5f}")

    df_cross = pd.DataFrame([
        {"Evaluation_Setting": "PhiUSIIL Test Split (In-Dataset)", **metrics_phi_common},
        {"Evaluation_Setting": "UCI External Dataset (Cross-Dataset)", **metrics_uci_external}
    ])
    cross_csv_path = os.path.join(results_dir, "cross_dataset_generalization.csv")
    df_cross.to_csv(cross_csv_path, index=False)
    print(f"Saved Cross-Dataset Generalization results to: {cross_csv_path}")

    # =========================================================================
    # PART C — URL-ONLY FEATURE SCOPE ABLATION (NO WEBPAGE-CRAWL FEATURES)
    # =========================================================================
    metrics_url_only = run_part_c(df_phi_clean, base_dir, results_dir, models_dir)

    # =========================================================================
    # CONSOLIDATED SUMMARY TABLE & SYNTHESIS
    # =========================================================================
    print("\n" + "=" * 85)
    print("CONSOLIDATED STAGE 2 SUMMARY TABLE")
    print("=" * 85)

    ann_kfold_acc = df_kfold[(df_kfold["Model"] == "Proposed ANN (30 MI)") & (df_kfold["Metric"] == "Accuracy")].iloc[0]
    ann_kfold_f1 = df_kfold[(df_kfold["Model"] == "Proposed ANN (30 MI)") & (df_kfold["Metric"] == "F1")].iloc[0]
    ann_kfold_auc = df_kfold[(df_kfold["Model"] == "Proposed ANN (30 MI)") & (df_kfold["Metric"] == "ROC-AUC")].iloc[0]

    consolidated_data = [
        {
            "Experimental_Evaluation": "Stage 1 Baseline: Proposed ANN (Single 80/20 Split, 30 MI Features)",
            "Accuracy": f"{stage1_metrics['Accuracy']*100:.3f}%",
            "F1_Score": f"{stage1_metrics['F1']*100:.3f}%",
            "ROC_AUC": f"{stage1_metrics['ROC-AUC']:.6f}",
            "Scope_and_Notes": "Full feature space (lexical + HTML/DOM), single split"
        },
        {
            "Experimental_Evaluation": "Part A: 5-Fold Stratified CV (Proposed ANN, Mean ± Std)",
            "Accuracy": f"{ann_kfold_acc['Mean']*100:.3f}% ± {ann_kfold_acc['Std Dev']*100:.4f}%",
            "F1_Score": f"{ann_kfold_f1['Mean']*100:.3f}% ± {ann_kfold_f1['Std Dev']*100:.4f}%",
            "ROC_AUC": f"{ann_kfold_auc['Mean']:.6f} ± {ann_kfold_auc['Std Dev']:.6f}",
            "Scope_and_Notes": "Full feature space, fresh per-fold scaling & MI selection"
        },
        {
            "Experimental_Evaluation": "Part B1: Cross-Dataset Common Features (PhiUSIIL Test Split)",
            "Accuracy": f"{metrics_phi_common['Accuracy']*100:.3f}%",
            "F1_Score": f"{metrics_phi_common['F1']*100:.3f}%",
            "ROC_AUC": f"{metrics_phi_common['ROC-AUC']:.6f}",
            "Scope_and_Notes": "8 common alignable features evaluated in-dataset"
        },
        {
            "Experimental_Evaluation": "Part B2: Cross-Dataset External Validation (UCI Dataset)",
            "Accuracy": f"{metrics_uci_external['Accuracy']*100:.3f}%",
            "F1_Score": f"{metrics_uci_external['F1']*100:.3f}%",
            "ROC_AUC": f"{metrics_uci_external['ROC-AUC']:.6f}",
            "Scope_and_Notes": "External generalization on Mohammad et al. (11,055 rows)"
        },
        {
            "Experimental_Evaluation": "Part C: URL-Only Scope Ablation (20 Pre-Crawl Features)",
            "Accuracy": f"{metrics_url_only['Accuracy']*100:.3f}%",
            "F1_Score": f"{metrics_url_only['F1']*100:.3f}%",
            "ROC_AUC": f"{metrics_url_only['ROC-AUC']:.6f}",
            "Scope_and_Notes": "Pre-crawl lexical/host features only, zero webpage fetching"
        }
    ]

    df_consolidated = pd.DataFrame(consolidated_data)
    consolidated_csv_path = os.path.join(results_dir, "stage2_consolidated_summary.csv")
    df_consolidated.to_csv(consolidated_csv_path, index=False)
    print(f"Saved Consolidated Summary to: {consolidated_csv_path}\n")

    for row in consolidated_data:
        print(f"[{row['Experimental_Evaluation']}]")
        print(f"  Accuracy: {row['Accuracy']} | F1: {row['F1_Score']} | ROC-AUC: {row['ROC_AUC']}")
        print(f"  Notes: {row['Scope_and_Notes']}\n")

    print("=" * 85)
    print("STAGE 2 PIPELINE EXECUTION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()
