#!/usr/bin/env python3
"""
FIFA World Cup 2026 Winner Prediction — Kaggle End-to-End Pipeline
===================================================================

This script provides a complete ML pipeline for predicting World Cup winners.
It includes data loading, preprocessing, feature engineering, model training,
evaluation, and submission file generation.

Compatible with Kaggle Notebooks.
"""

# =============================================================================
# STEP 1 — IMPORTS
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb

# Optional (uncomment if available on Kaggle):
# import lightgbm as lgb
# from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 2 — LOAD DATA
# =============================================================================

# On Kaggle, adjust paths to: /kaggle/input/<dataset-name>/
TRAIN_PATH = "/home/z/my-project/download/train.csv"
TEST_PATH = "/home/z/my-project/download/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"\nTrain columns:\n{list(train.columns)}")
print(f"\nWinner distribution:\n{train['winner'].value_counts()}")
print(f"\nMissing values (train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")


# =============================================================================
# STEP 3 — PREPROCESSING
# =============================================================================

target = "winner"

# Separate features
features = [col for col in train.columns if col != target and train[col].dtype != "object"]
print(f"\nNumeric features ({len(features)}): {features}")

X = train[features].copy()
y = train[target].copy()
X_test = test[features].copy()

# Fill missing values with column mean
X = X.fillna(X.mean())
X_test = X_test.fillna(X_test.mean())


# =============================================================================
# STEP 4 — FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """Create derived features to boost model performance."""
    df = df.copy()

    # FIFA Strength Index: composite of points, form, and rating
    df["strength_index"] = (
        df["fifa_points"] +
        df["recent_form_score"] * 50 +
        df["avg_player_rating"] * 10
    )

    # Goal efficiency: scored vs conceded ratio
    df["goal_efficiency"] = df["goals_scored_avg"] / df["goals_conceded_avg"].clip(lower=0.1)

    # Attack potency: shots on target * scoring rate
    df["attack_potency"] = df["shots_per_game"] * df["shots_on_target_ratio"] * df["goals_scored_avg"]

    # Defensive solidity: clean sheets + low concession
    df["defensive_solidity"] = df["clean_sheets_last_10"] / (df["goals_conceded_avg"] + 0.5)

    # Squad quality index: rating * market value * experience
    df["squad_quality"] = (
        df["avg_player_rating"] * 0.4 +
        (df["market_value_million_eur"] / 1200) * 100 * 0.3 +
        (df["experience_avg_caps"] / 65) * 100 * 0.3
    )

    # Form consistency: win rate adjusted by form score
    df["form_consistency"] = df["win_rate_last_year"] * df["recent_form_score"]

    # Possession dominance: possession * passing accuracy
    df["possession_dominance"] = df["possession_avg"] * df["passing_accuracy"] / 100

    # Star power: star players * market value
    df["star_power"] = df["star_players_count"] * df["market_value_million_eur"]

    # Home advantage boost
    df["contextual_advantage"] = (
        df["host_advantage"] * 10 +
        df["climate_similarity_score"] * 5 -
        df["travel_distance_avg"] * 0.5
    )

    return df

X = engineer_features(X)
X_test = engineer_features(X_test)

# Update feature list
all_features = X.columns.tolist()
print(f"\nFeatures after engineering: {len(all_features)}")


# =============================================================================
# STEP 5 — TRAIN/VALIDATION SPLIT
# =============================================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain split: {X_train.shape}")
print(f"Val split:   {X_val.shape}")


# =============================================================================
# STEP 6 — MODEL TRAINING (XGBOOST)
# =============================================================================

print("\n" + "=" * 60)
print("Training XGBoost Classifier...")
print("=" * 60)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    gamma=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)


# =============================================================================
# STEP 7 — EVALUATION
# =============================================================================

val_pred = xgb_model.predict(X_val)
val_proba = xgb_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, val_pred)
auc = roc_auc_score(y_val, val_proba)

print(f"\n{'='*60}")
print(f"VALIDATION RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC:  {auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_val, val_pred, target_names=["Not Winner", "Winner"]))


# =============================================================================
# STEP 8 — FEATURE IMPORTANCE
# =============================================================================

importance = pd.DataFrame({
    "feature": all_features,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print(f"\nTop 15 Features:")
print(importance.head(15).to_string(index=False))


# =============================================================================
# STEP 9 — OPTIONAL: ENSEMBLE (XGBoost + RandomForest)
# =============================================================================

print("\n" + "=" * 60)
print("Training Random Forest for Ensemble...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_val_proba)
print(f"Random Forest AUC: {rf_auc:.4f}")

# Blended ensemble
ensemble_proba = 0.6 * val_proba + 0.4 * rf_val_proba
ensemble_auc = roc_auc_score(y_val, ensemble_proba)
print(f"Ensemble AUC (0.6*XGB + 0.4*RF): {ensemble_auc:.4f}")


# =============================================================================
# STEP 10 — PROBABILITY CALIBRATION
# =============================================================================

print("\nCalibrating probabilities...")
calibrated_model = CalibratedClassifierCV(xgb_model, cv=3, method='isotonic')
calibrated_model.fit(X_train, y_train)

cal_val_proba = calibrated_model.predict_proba(X_val)[:, 1]
cal_auc = roc_auc_score(y_val, cal_val_proba)
print(f"Calibrated XGBoost AUC: {cal_auc:.4f}")


# =============================================================================
# STEP 11 — TRAIN ON FULL DATA & PREDICT TEST
# =============================================================================

print("\n" + "=" * 60)
print("Training final model on full data...")
print("=" * 60)

final_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    gamma=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
)

final_model.fit(X, y)

# Test predictions
test_proba_xgb = final_model.predict_proba(X_test)[:, 1]

# Also train RF on full data
rf_full = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_full.fit(X, y)
test_proba_rf = rf_full.predict_proba(X_test)[:, 1]

# Ensemble
test_proba_final = 0.6 * test_proba_xgb + 0.4 * test_proba_rf


# =============================================================================
# STEP 12 — SUBMISSION FILE
# =============================================================================

submission = pd.DataFrame({
    "id": range(len(test_proba_final)),
    "winner_probability": test_proba_final
})

submission_path = "/home/z/my-project/download/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"\nSubmission saved to: {submission_path}")
print(f"Submission shape: {submission.shape}")
print(f"\nSubmission preview:")
print(submission.head(20).to_string(index=False))
print(f"\nProbability stats:")
print(submission["winner_probability"].describe())

# Top predicted winners
test_with_pred = test.copy()
test_with_pred["winner_probability"] = test_proba_final

top_teams = test_with_pred.groupby("team_name")["winner_probability"].mean().sort_values(ascending=False)
print(f"\n{'='*60}")
print("TOP 10 PREDICTED WORLD CUP WINNERS")
print(f"{'='*60}")
for i, (team, prob) in enumerate(top_teams.head(10).items(), 1):
    print(f"  {i:2d}. {team:20s}  Win Probability: {prob:.4f}")


# =============================================================================
# STEP 13 — CROSS-VALIDATION SCORE
# =============================================================================

print(f"\n{'='*60}")
print("5-Fold Cross-Validation")
print(f"{'='*60}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1, use_label_encoder=False,
    )
    fold_model.fit(X_fold_train, y_fold_train)
    fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, fold_proba)
    cv_scores.append(fold_auc)
    print(f"  Fold {fold}: AUC = {fold_auc:.4f}")

print(f"\n  Mean AUC: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
