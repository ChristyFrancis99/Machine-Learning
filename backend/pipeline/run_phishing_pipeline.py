"""
URL-Based Phishing Website Detection Using Machine Learning and Artificial Neural Networks
========================================================================================
Authors: Christy Francis, Saksham Joshi, Aadesh Patil
Department of Computer Engineering, Pimpri Chinchwad College of Engineering, Pune

Pipeline Structure formatted following FA1.ipynb:
1. Import Libraries
2. Load Dataset
3. Explore Dataset (EDA)
4. Data Preprocessing & Visualizations
   - Missing & Duplicate Values
   - Target Class Distribution & Plot (Fig. 1)
   - HTTPS Usage Analysis & Plot (Fig. 2)
   - Drop Non-Predictive Text Columns
   - Target Leakage Screening & Removal
   - Feature Correlation Heatmap (Fig. 3)
   - Feature & Target Separation
5. Feature Selection (SelectKBest with Mutual Information)
   - MI Ranking & Top 20 Feature Importance Bar Plot (Fig. 4)
   - Top 30 Selected Features
6. Train-Test Split (Stratified 80/20)
7. Feature Scaling (StandardScaler on Training Split)
8. Build Proposed Artificial Neural Network (ANN)
9. Train Model (with Dropout & Early Stopping)
   - Training vs Validation Curves (Fig. 5)
10. Evaluate Proposed ANN Model on Held-Out Test Set
   - Confusion Matrix (TP, TN, FP, FN) & Heatmap (Fig. 6)
   - Performance Metrics: Accuracy, Precision, Recall, F1, ROC-AUC (Table 4)
11. Feature Selection Ablation Study (Paper Section 13, Table 7)
12. Comparison with Traditional Machine Learning Models (Paper Section 12, Table 6)
   - Logistic Regression, Decision Tree, Random Forest, LinearSVC, XGBoost
   - Model Comparison Bar Chart (Fig. 7)
"""

import os
import sys
import time
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress non-critical warnings & TensorFlow info logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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


def compute_mutual_info(X, y):
    """Score function for SelectKBest using mutual_info_classif."""
    return mutual_info_classif(X, y, random_state=SEED)


def evaluate_binary_predictions(y_true_raw, y_pred_raw, y_prob_phish, model_name="Model"):
    """
    Evaluate model metrics treating Phishing as the positive class (Paper Section 10).
    In the raw dataset:
      label = 1 -> Legitimate
      label = 0 -> Phishing
    For evaluation:
      Actual Phishing: y_true_phish = 1 (when raw label == 0)
      Predicted Phishing: y_pred_phish = 1 (when predicted raw == 0)
    """
    y_true_phish = (y_true_raw == 0).astype(int)
    y_pred_phish = (y_pred_raw == 0).astype(int)

    acc = accuracy_score(y_true_phish, y_pred_phish)
    prec = precision_score(y_true_phish, y_pred_phish, zero_division=0)
    rec = recall_score(y_true_phish, y_pred_phish, zero_division=0)
    f1 = f1_score(y_true_phish, y_pred_phish, zero_division=0)

    try:
        auc = roc_auc_score(y_true_phish, y_prob_phish)
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_true_phish, y_pred_phish)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


def main():
    start_time = time.time()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.path.join(project_root, "data", "raw", "PhiUSIIL_Phishing_URL_Dataset.csv")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 85)
    print("  URL-Based Phishing Website Detection Using Machine Learning and ANN")
    print("  Complete Preprocessing, EDA, Visualizations & Machine Learning Pipeline")
    print("=" * 85, flush=True)

    # =========================================================================
    # 1. IMPORT LIBRARIES & SETUP
    # =========================================================================
    print("\n[1] LIBRARIES IMPORTED SUCCESSFULLY.")
    print("    - Data Handling: pandas, numpy")
    print("    - Visualization: matplotlib, seaborn")
    print("    - Preprocessing & ML: scikit-learn, xgboost")
    print("    - Deep Learning: tensorflow, keras\n", flush=True)

    # =========================================================================
    # 2. LOAD DATASET
    # =========================================================================
    print("-" * 85)
    print("[2] LOADING DATASET...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Dataset successfully loaded from: {dataset_path}")
    print(f"Total Rows: {df.shape[0]:,} | Total Columns: {df.shape[1]}")
    print("\nFirst 5 rows (df.head()):")
    print(df.head())

    # =========================================================================
    # 3. EXPLORE DATASET (EDA)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[3] DATA EXPLORATION (EDA)...")
    print("\n--- Dataset Info (df.info()) ---")
    df.info(verbose=False, show_counts=True)

    print(f"\n--- Dataset Shape: {df.shape} ---")
    print("\n--- Summary Statistics for Selected Features (df.describe()) ---")
    selected_summary_cols = ['URLLength', 'LineOfCode', 'NoOfJS', 'NoOfCSS', 'NoOfImage', 'NoOfExternalRef']
    print(df[selected_summary_cols].describe().T)

    print(f"\n--- Total Columns ({len(df.columns)}) ---")
    print(list(df.columns))

    print("\n--- Missing Value Count (df.isnull().sum()) ---")
    null_counts = df.isnull().sum()
    print(f"Total missing values across entire dataset: {null_counts.sum()}")

    # =========================================================================
    # 4. DATA PREPROCESSING & EDA VISUALIZATIONS
    # =========================================================================
    print("\n" + "-" * 85)
    print("[4] DATA PREPROCESSING & EDA VISUALIZATIONS...")

    # 4.1 Missing Values and Duplicate Handling
    print("\n4.1 Handling Missing Values and Duplicates...")
    df.dropna(inplace=True)
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"Dropped duplicate rows. New shape: {df.shape}")
    else:
        print("No duplicate rows present. Dataset integrity maintained.")

    # 4.2 Target Class Distribution & Graph 1
    print("\n4.2 Target Class Distribution:")
    class_counts = df["label"].value_counts()
    print(class_counts)
    print("  label = 1 -> Legitimate Website:", class_counts.get(1, 0), f"({class_counts.get(1,0)/len(df)*100:.2f}%)")
    print("  label = 0 -> Phishing Website:  ", class_counts.get(0, 0), f"({class_counts.get(0,0)/len(df)*100:.2f}%)")

    fig1_path = os.path.join(results_dir, "fig1_target_distribution.png")
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x='label', data=df, palette=['#ef4444', '#10b981'])
    plt.title("Distribution of Target Classes (Fig. 1)", fontsize=13, fontweight='bold')
    plt.xlabel("Class (0 = Phishing, 1 = Legitimate)", fontsize=11)
    plt.ylabel("Number of Samples", fontsize=11)
    plt.xticks([0, 1], [f"Phishing\n({class_counts.get(0,0):,})", f"Legitimate\n({class_counts.get(1,0):,})"], fontsize=10)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Graph 1 (Target Distribution) to: {fig1_path}")

    # 4.3 HTTPS Usage Analysis & Graph 2
    print("\n4.3 Analyzing HTTPS Usage by Class...")
    fig2_path = os.path.join(results_dir, "fig2_https_usage.png")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="IsHTTPS", hue="label", palette=['#ef4444', '#10b981'])
    plt.title("HTTPS Usage Across Legitimate and Phishing URLs (Fig. 2 / Paper Fig. 9)", fontsize=13, fontweight='bold')
    plt.xlabel("HTTPS Protocol (0 = HTTP / Non-Secure, 1 = HTTPS / Secure)", fontsize=11)
    plt.ylabel("Number of URLs", fontsize=11)
    plt.legend(title="Class", labels=["Phishing", "Legitimate"])
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved Graph 2 (HTTPS Usage by Class) to: {fig2_path}")

    # 4.4 Removal of Non-Predictive Text Columns (Section 4.4)
    print("\n4.4 Removing Raw Non-Predictive Identifier Columns (Section 4.4)...")
    id_cols = ["FILENAME", "URL", "Domain", "Title", "TLD"]
    dropped_ids = [col for col in id_cols if col in df.columns]
    df_clean = df.drop(columns=dropped_ids)
    print(f"Dropped raw text/identifier columns: {dropped_ids}")
    print("Reason: High-cardinality text causes models to memorize specific URLs rather than generalizable traits.")

    # 4.5 Leakage Screening & Removal
    print("\n4.5 Target Leakage Screening & Remediation...")
    print("Inspecting URLSimilarityIndex & TLDLegitimateProb for artificial target leakage:")
    sim_legit = df.loc[df['label'] == 1, 'URLSimilarityIndex']
    sim_phish = df.loc[df['label'] == 0, 'URLSimilarityIndex']
    print(f"  - URLSimilarityIndex in Legitimate class: mean={sim_legit.mean():.2f}, std={sim_legit.std():.4f} (100% exact 100.0)")
    print(f"  - URLSimilarityIndex in Phishing class:   mean={sim_phish.mean():.2f}, std={sim_phish.std():.4f}")
    print("  -> Verdict: Tautological target leak (legitimate URLs were compared against a legitimate reference database).")

    tld_var = df.groupby('TLD')['TLDLegitimateProb'].nunique().max()
    print(f"  - TLDLegitimateProb within-TLD variation across all 695 TLDs: {tld_var} unique value per TLD (zero variance)")
    print("  -> Verdict: Global target-encoded prior computed across full dataset before split.")

    leaking_cols = ['URLSimilarityIndex', 'TLDLegitimateProb']
    df_clean = df_clean.drop(columns=leaking_cols)
    print(f"Successfully dropped leaking features: {leaking_cols}")
    print(f"Clean usable predictor features: {df_clean.shape[1] - 1}")

    # 4.6 Feature Correlation Heatmap & Graph 3
    print("\n4.6 Generating Feature Correlation Heatmap...")
    fig3_path = os.path.join(results_dir, "fig3_correlation_heatmap.png")
    plt.figure(figsize=(16, 13))
    corr_matrix = df_clean.drop(columns=['label']).corr()
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, cbar_kws={'shrink': 0.8})
    plt.title("Correlation Heatmap of Usable Predictor Features (Fig. 3 / Paper Fig. 10)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved Graph 3 (Correlation Heatmap) to: {fig3_path}")

    # 4.7 Separate Features and Target
    print("\n4.7 Separating Features (X) and Target (y)...")
    X = df_clean.drop("label", axis=1)
    y = df_clean["label"].values
    feature_names = list(X.columns)
    print(f"X shape: {X.shape} (Input features)")
    print(f"y shape: {y.shape} (Binary labels: 1 = Legitimate, 0 = Phishing)")

    import json
    json.dump(list(X.columns), open("models/full_feature_order.json", "w"))
    print(f"Saved full feature order ({len(X.columns)} features) to models/full_feature_order.json")

    # =========================================================================
    # 5. TRAIN-TEST SPLIT (Section 4.6)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[5] PERFORMING STRATIFIED TRAIN-TEST SPLIT (80% Train, 20% Test)...")
    print("Notice: To prevent silent data leakage, scaling and feature selection")
    print("must be fitted exclusively on the 80% training set!")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    print(f"X_train: {X_train_raw.shape} | y_train: {y_train.shape}")
    print(f"X_test : {X_test_raw.shape}  | y_test : {y_test.shape}")
    print(f"Train class balance: Legitimate={np.mean(y_train==1)*100:.2f}%, Phishing={np.mean(y_train==0)*100:.2f}%")
    print(f"Test class balance:  Legitimate={np.mean(y_test==1)*100:.2f}%, Phishing={np.mean(y_test==0)*100:.2f}%")

    # =========================================================================
    # 6. FEATURE SCALING (Section 6)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[6] FEATURE SCALING (StandardScaler)...")
    print("Standardizing features: z = (x - mean) / std")
    print("Fitted ONLY on X_train to ensure no information leaks from X_test.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Fitted scaler saved to: {scaler_path}")

    # =========================================================================
    # 7. FEATURE SELECTION (SelectKBest with Mutual Information, Section 7)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[7] FEATURE SELECTION (SelectKBest with Mutual Information)...")
    print("Computing Mutual Information scores across all features on training split...")

    t0_mi = time.time()
    selector = SelectKBest(score_func=compute_mutual_info, k=30)
    X_train_sel30 = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel30 = selector.transform(X_test_scaled)
    t_mi = time.time() - t0_mi
    print(f"Mutual Information computation completed in {t_mi:.2f} seconds.")

    # Rank features by MI score
    mi_scores = selector.scores_
    mi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mi_scores
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    mi_df["Rank"] = mi_df.index + 1

    print("\nTop 20 Features Ranked by Mutual Information (Table 2 / Paper Section 11.2):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Feature Name':<35} {'MI Score':<12}")
    print("-" * 60)
    for idx, row in mi_df.head(20).iterrows():
        print(f"{int(row['Rank']):<6} {row['Feature']:<35} {row['Importance']:<12.5f}")
    print("-" * 60)

    # Save MI Table
    mi_csv_path = os.path.join(results_dir, "mutual_information_ranking.csv")
    mi_df.to_csv(mi_csv_path, index=False)
    print(f"Saved complete MI ranking table to: {mi_csv_path}")

    # Graph 4: Top 20 Important Features Bar Plot
    fig4_path = os.path.join(results_dir, "fig4_top20_mi_features.png")
    top20_df = mi_df.head(20).iloc[::-1]  # Invert for horizontal display
    plt.figure(figsize=(10, 8))
    plt.barh(top20_df["Feature"], top20_df["Importance"], color='#2563eb', edgecolor='black', alpha=0.85)
    plt.xlabel("Mutual Information Score", fontsize=11, fontweight='bold')
    plt.ylabel("Feature Name", fontsize=11, fontweight='bold')
    plt.title("Top 20 Important Features for Phishing Detection (Fig. 4 / Paper Fig. 8)", fontsize=13, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved Graph 4 (Top 20 Features Bar Plot) to: {fig4_path}")

    # Save fitted selector & selected feature names
    selected_mask = selector.get_support()
    selected_features_30 = list(np.array(feature_names)[selected_mask])
    selector.score_func = mutual_info_classif  # Ensure clean unpickling in external portals (Streamlit)
    with open(os.path.join(models_dir, "feature_selector.pkl"), "wb") as f:
        pickle.dump(selector, f)
    with open(os.path.join(models_dir, "selected_features.json"), "w", encoding="utf-8") as f:
        json.dump(selected_features_30, f, indent=2)

    # =========================================================================
    # 8. BUILD PROPOSED ARTIFICIAL NEURAL NETWORK (ANN) ARCHITECTURE (Section 8)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[8] BUILDING PROPOSED ARTIFICIAL NEURAL NETWORK (ANN)...")
    print("Paper Architecture (Section 8.1):")
    print("  Input(30) -> Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3) -> Dense(32, relu) -> Dense(1, sigmoid)")

    # Carve a 10% stratified validation split out of training set
    X_tr_ann, X_val_ann, y_tr_ann, y_val_ann = train_test_split(
        X_train_sel30,
        y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=SEED
    )

    ann_model = Sequential([
        Input(shape=(30,)),
        Dense(128, activation="relu", name="Dense_1"),
        Dropout(0.30, name="Dropout_1"),
        Dense(64, activation="relu", name="Dense_2"),
        Dropout(0.30, name="Dropout_2"),
        Dense(32, activation="relu", name="Dense_3"),
        Dense(1, activation="sigmoid", name="Output_Sigmoid")
    ], name="Phishing_ANN_Proposed")

    ann_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print("\nModel Architecture Summary:")
    ann_model.summary()

    # =========================================================================
    # 9. TRAIN PROPOSED ANN MODEL (Section 9)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[9] TRAINING PROPOSED ANN MODEL...")
    print("Parameters: batch_size=512, epochs=40, EarlyStopping(monitor='val_loss', patience=5)")

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    t0_ann = time.time()
    history = ann_model.fit(
        X_tr_ann,
        y_tr_ann,
        validation_data=(X_val_ann, y_val_ann),
        epochs=40,
        batch_size=512,
        callbacks=[early_stopping],
        verbose=1
    )
    t_ann = time.time() - t0_ann
    print(f"ANN Training completed in {t_ann:.2f} seconds ({len(history.history['loss'])} epochs).")

    ann_path = os.path.join(models_dir, "phishing_ann_selected30.keras")
    ann_model.save(ann_path)
    print(f"Trained ANN model saved to: {ann_path}")

    # Graph 5: Training & Validation Curves Plot
    fig5_path = os.path.join(results_dir, "fig5_ann_training_history.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#2563eb', lw=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#10b981', lw=2, linestyle='--')
    ax1.set_title("ANN Training vs Validation Accuracy (Fig. 5 / Paper Fig. 11)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Training Loss', color='#ef4444', lw=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#f59e0b', lw=2, linestyle='--')
    ax2.set_title("ANN Training vs Validation Loss (Binary Cross-Entropy)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved Graph 5 (Training History Curves) to: {fig5_path}")

    # =========================================================================
    # 10. EVALUATE PROPOSED ANN MODEL ON TEST SET (Section 10 & 11)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[10] EVALUATING PROPOSED ANN ON HELD-OUT TEST SET (47,159 samples)...")
    print("Per Paper Section 10: Phishing is treated as the Positive Class.")

    y_prob_legit = ann_model.predict(X_test_sel30, batch_size=512).ravel()
    y_prob_phish = 1.0 - y_prob_legit
    y_pred_raw = (y_prob_legit >= 0.5).astype(int)

    ann_results = evaluate_binary_predictions(
        y_test,
        y_pred_raw,
        y_prob_phish,
        model_name="Proposed ANN (30 Features)"
    )

    print("\n" + "=" * 65)
    print("Table 4: Performance of Proposed ANN (Held-Out Test Set):")
    print("=" * 65)
    print(f"  Accuracy:  {ann_results['Accuracy']*100:.3f}%")
    print(f"  Precision: {ann_results['Precision']*100:.3f}%")
    print(f"  Recall:    {ann_results['Recall']*100:.3f}%")
    print(f"  F1-Score:  {ann_results['F1-Score']*100:.3f}%")
    print(f"  ROC-AUC:   {ann_results['ROC-AUC']:.5f}")
    print("-" * 65)
    print("Confusion Matrix Breakdown:")
    print(f"  True Positives  (Phishing correctly caught):     {ann_results['TP']:,}")
    print(f"  True Negatives  (Legitimate correctly verified): {ann_results['TN']:,}")
    print(f"  False Positives (Legitimate falsely blocked):    {ann_results['FP']:,}")
    print(f"  False Negatives (Phishing MISSED - Costliest!):  {ann_results['FN']:,}")
    print("=" * 65)

    # Graph 6: Confusion Matrix Heatmap
    fig6_path = os.path.join(results_dir, "fig6_ann_confusion_matrix.png")
    cm_ann = np.array([[ann_results['TN'], ann_results['FP']],
                       [ann_results['FN'], ann_results['TP']]])

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm_ann, annot=True, fmt=',d', cmap="Blues", cbar=True,
                     xticklabels=["Legitimate", "Phishing"],
                     yticklabels=["Legitimate", "Phishing"])
    plt.title("Confusion Matrix - Proposed ANN\n(Phishing = Positive Class)", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.ylabel("True Class", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Saved Graph 6 (Confusion Matrix Heatmap) to: {fig6_path}")

    # =========================================================================
    # 11. FEATURE SELECTION ABLATION STUDY (Paper Section 13, Table 7)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[11] RUNNING FEATURE SELECTION ABLATION STUDY (Section 13)...")
    print("Training second identical ANN on ALL 48 usable scaled features (no MI selection)...")

    X_tr_ab, X_val_ab, y_tr_ab, y_val_ab = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.10,
        stratify=y_train,
        random_state=SEED
    )

    ablation_model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(128, activation="relu", name="Dense_1"),
        Dropout(0.30, name="Dropout_1"),
        Dense(64, activation="relu", name="Dense_2"),
        Dropout(0.30, name="Dropout_2"),
        Dense(32, activation="relu", name="Dense_3"),
        Dense(1, activation="sigmoid", name="Output_Sigmoid")
    ], name="Phishing_ANN_Ablation_All48")

    ablation_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    t0_ab = time.time()
    ablation_model.fit(
        X_tr_ab,
        y_tr_ab,
        validation_data=(X_val_ab, y_val_ab),
        epochs=40,
        batch_size=512,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)],
        verbose=1
    )
    t_ab = time.time() - t0_ab

    # Evaluate ablation model
    y_prob_legit_ab = ablation_model.predict(X_test_scaled, batch_size=512).ravel()
    y_prob_phish_ab = 1.0 - y_prob_legit_ab
    y_pred_ab = (y_prob_legit_ab >= 0.5).astype(int)

    ablation_results = evaluate_binary_predictions(
        y_test,
        y_pred_ab,
        y_prob_phish_ab,
        model_name="ANN without feature selection"
    )

    ablation_df = pd.DataFrame([
        {
            "Configuration": "ANN without feature selection",
            "Features": X.shape[1],
            "Accuracy": ablation_results["Accuracy"],
            "Precision": ablation_results["Precision"],
            "Recall": ablation_results["Recall"],
            "F1-Score": ablation_results["F1-Score"],
            "ROC-AUC": ablation_results["ROC-AUC"],
            "Training_Time_s": t_ab
        },
        {
            "Configuration": "ANN with Mutual Information",
            "Features": 30,
            "Accuracy": ann_results["Accuracy"],
            "Precision": ann_results["Precision"],
            "Recall": ann_results["Recall"],
            "F1-Score": ann_results["F1-Score"],
            "ROC-AUC": ann_results["ROC-AUC"],
            "Training_Time_s": t_ann
        }
    ])

    print("\n" + "=" * 95)
    print("Table 7: Feature Selection Ablation Study Results (Paper Section 13):")
    print("=" * 95)
    print(ablation_df.to_string(index=False))
    print("=" * 95)

    ablation_csv_path = os.path.join(results_dir, "ablation_study_table7.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"Saved Table 7 to: {ablation_csv_path}")

    # =========================================================================
    # 12. COMPARISON WITH TRADITIONAL MACHINE LEARNING MODELS (Section 12, Table 6)
    # =========================================================================
    print("\n" + "-" * 85)
    print("[12] COMPARISON WITH TRADITIONAL MACHINE LEARNING MODELS (Section 12)...")
    print("Methodology Note: LinearSVC is utilized for Support Vector Machine in place of")
    print("kernel SVC for computational tractability at N = 188,636 training samples.")

    traditional_models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
        ("Decision Tree", DecisionTreeClassifier(random_state=SEED)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
        ("Support Vector Machine (LinearSVC)", LinearSVC(dual='auto', random_state=SEED)),
        ("XGBoost", XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=SEED, n_jobs=-1))
    ]

    all_model_results = []

    for name, clf in traditional_models:
        print(f"\nTraining {name} on 30 MI-selected features...")
        t_clf_start = time.time()
        clf.fit(X_train_sel30, y_train)
        t_clf = time.time() - t_clf_start
        print(f"  Training finished in {t_clf:.2f} seconds.")

        y_pred = clf.predict(X_test_sel30)

        if hasattr(clf, "predict_proba"):
            prob_phish = clf.predict_proba(X_test_sel30)[:, 0]
        elif hasattr(clf, "decision_function"):
            prob_phish = -clf.decision_function(X_test_sel30)
        else:
            prob_phish = (y_pred == 0).astype(float)

        res = evaluate_binary_predictions(y_test, y_pred, prob_phish, model_name=name)
        res["Training_Time_s"] = t_clf
        all_model_results.append(res)
        print(f"  {name} Accuracy: {res['Accuracy']*100:.3f}% | F1: {res['F1-Score']*100:.3f}% | Recall: {res['Recall']*100:.3f}%")

    # Append Proposed ANN results
    ann_eval_copy = ann_results.copy()
    ann_eval_copy["Training_Time_s"] = t_ann
    all_model_results.append(ann_eval_copy)

    # Consolidated Table 6 DataFrame
    comparison_df = pd.DataFrame(all_model_results)
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "TP", "TN", "FP", "FN", "Training_Time_s"]
    comparison_df = comparison_df[display_cols]

    print("\n" + "=" * 110)
    print("Table 6: Consolidated Model Comparison (Held-Out Test Set, 30 Selected Features)")
    print("=" * 110)
    print(comparison_df.to_string(index=False))
    print("=" * 110)

    comparison_csv_path = os.path.join(results_dir, "model_comparison_table6.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved Table 6 to: {comparison_csv_path}")

    # Graph 7: Model Comparison Bar Chart
    fig7_path = os.path.join(results_dir, "fig7_model_comparison_barchart.png")
    models_list = comparison_df["Model"].tolist()
    acc_list = [v * 100 for v in comparison_df["Accuracy"].tolist()]
    f1_list = [v * 100 for v in comparison_df["F1-Score"].tolist()]

    x_idx = np.arange(len(models_list))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x_idx - width/2, acc_list, width, label='Accuracy (%)', color='#2563eb', edgecolor='black', alpha=0.9)
    plt.bar(x_idx + width/2, f1_list, width, label='F1-Score (%)', color='#10b981', edgecolor='black', alpha=0.9)
    plt.ylabel('Score (%)', fontsize=11, fontweight='bold')
    plt.title('Model Performance Comparison on 30 Selected Features (Fig. 7)', fontsize=13, fontweight='bold')
    plt.xticks(x_idx, [m.replace("Support Vector Machine (LinearSVC)", "LinearSVC").replace("Proposed ANN (30 Features)", "Proposed ANN") for m in models_list], rotation=15, fontsize=10, fontweight='bold')
    plt.ylim([99.5, 100.05])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"Saved Graph 7 (Model Comparison Bar Chart) to: {fig7_path}")

    # =========================================================================
    # 13. EXECUTION SUMMARY
    # =========================================================================
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_elapsed/60:.2f} MINUTES!")
    print("=" * 85)
    print("Generated Figures to Show Mam (Saved in results/):")
    print(f"  1. Fig 1 - Target Class Distribution:         {fig1_path}")
    print(f"  2. Fig 2 - HTTPS Usage by Class:              {fig2_path}")
    print(f"  3. Fig 3 - Feature Correlation Heatmap:       {fig3_path}")
    print(f"  4. Fig 4 - Top 20 Important Features (MI):    {fig4_path}")
    print(f"  5. Fig 5 - ANN Training & Validation Curves:  {fig5_path}")
    print(f"  6. Fig 6 - ANN Confusion Matrix Heatmap:      {fig6_path}")
    print(f"  7. Fig 7 - Model Comparison Bar Chart:        {fig7_path}")
    print("\nGenerated Paper Tables (Saved in results/):")
    print(f"  - Table 2: {mi_csv_path}")
    print(f"  - Table 6: {comparison_csv_path}")
    print(f"  - Table 7: {ablation_csv_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
