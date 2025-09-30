#!/usr/bin/env python3
# train_and_evaluate_models.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, matthews_corrcoef, cohen_kappa_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks, backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="IaC conflict prediction with RF (window-agg) and LSTM (sequence)")
    ap.add_argument("--csv", required=True, help="Path to CSV with event-level rows")
    ap.add_argument("--window", type=int, default=32, help="Sliding window length (timesteps)")
    ap.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds (>=2)")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for LSTM")
    ap.add_argument("--epochs", type=int, default=150, help="Max epochs for LSTM")
    ap.add_argument("--val_frac", type=float, default=0.15, help="Fraction of train (within fold) used for validation")
    ap.add_argument("--plots_prefix", default="cv", help="Prefix for saved figures (last fold)")
    return ap.parse_args()


# =========================
# Metrics & helpers
# =========================
def f1_metric(y_true, y_pred):
    # Correct, non-broadcasting F1 (metric only; not a loss)
    y_true = K.flatten(K.cast(y_true, "float32"))
    y_pred_bin = K.flatten(K.round(y_pred))
    tp = K.sum(y_true * y_pred_bin)
    pp = K.sum(y_pred_bin)
    ppos = K.sum(y_true)
    precision = tp / (pp + K.epsilon())
    recall = tp / (ppos + K.epsilon())
    return 2.0 * (precision * recall) / (precision + recall + K.epsilon())


def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# =========================
# Feature engineering
# =========================
def build_basic_features(df):
    # Label encode categoricals
    le_res = LabelEncoder()
    le_attr = LabelEncoder()
    le_user = LabelEncoder()
    df["resource_id_encoded"] = le_res.fit_transform(df["resource_id"].astype(str))
    df["attribute_encoded"]  = le_attr.fit_transform(df["attribute"].astype(str))
    df["user_role_encoded"]  = le_user.fit_transform(df["user_role"].astype(str))

    # Time → seconds
    mt = pd.to_datetime(df["modification_time"], errors="coerce")
    if mt.isna().any():
        raise ValueError("Some 'modification_time' failed to parse")
    # int64 ns → seconds
    df["modification_time_numerical"] = (mt.view("int64") / 1e9).astype("float64")

    # If change_frequency missing, create a simple per-resource rolling count
    if "change_frequency" not in df.columns:
        df = df.sort_values(["resource_id", "modification_time"]).copy()
        df["change_frequency"] = (
            df.groupby("resource_id")["attribute"]
              .transform(lambda s: s.rolling(window=5, min_periods=1).count())
              .astype(float)
        )

    # Sort and derive sequential helpers
    df = df.sort_values(["resource_id", "modification_time"]).copy()
    df["time_since_last_change"] = df.groupby("resource_id")["modification_time_numerical"].diff().fillna(0.0)
    df["concurrent_count_3"] = (
        df.groupby("resource_id")["is_concurrent"].transform(
            lambda s: s.astype(float).rolling(window=3, min_periods=1).sum().fillna(0.0)
        )
    )
    df["change_frequency_diff"] = df.groupby("resource_id")["change_frequency"].diff().fillna(0.0)

    # Ensure numeric dtypes for all features used downstream
    feat_cols = [
        "resource_id_encoded","attribute_encoded","change_frequency",
        "modification_time_numerical","is_concurrent","user_role_encoded",
        "criticality_score","attribute_sensitivity_score","change_magnitude",
        "time_since_last_change","concurrent_count_3","change_frequency_diff"
    ]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df


def window_sequences(df, window=32, stride=8):
    """
    Build sliding-window sequences per resource. Each window yields:
      - X_seq: [T, F] sequence of timestep features
      - y: label (here, we take the label at the *last* timestep of the window)
      - seq_feat: sequence-level features (total concurrent, avg delta, last3 concurrent)
    Returns:
      X_seqs (list of float32 ndarrays [T,F]), y (int32 array), seq_feats (float32 array), groups (object array of resource ids)
    """
    feat_cols = [
        "resource_id_encoded","attribute_encoded","change_frequency",
        "modification_time_numerical","is_concurrent","user_role_encoded",
        "criticality_score","attribute_sensitivity_score","change_magnitude",
        "time_since_last_change","concurrent_count_3","change_frequency_diff"
    ]
    X_seqs, y_list, seq_feats, groups = [], [], [], []
    for rid, g in df.groupby("resource_id", sort=False):
        g = g.sort_values("modification_time_numerical")
        arr = g[feat_cols].astype("float32").values
        labels = g["conflict_after_sequence"].astype(int).values

        is_conc = g["is_concurrent"].to_numpy()
        times = g["modification_time_numerical"].to_numpy()

        for start in range(0, max(1, len(arr) - window + 1), stride):
            end = start + window
            if end <= len(arr):
                win = arr[start:end]
                X_seqs.append(np.asarray(win, dtype=np.float32))  # keep as numeric 2-D
                y_list.append(int(labels[end - 1]))

                wnd_is_conc = is_conc[start:end]
                wnd_times = times[start:end]
                total_conc = float(wnd_is_conc.sum())
                deltas = np.diff(wnd_times)
                avg_dt = float(deltas.mean()) if deltas.size > 0 else 0.0
                last3 = float(wnd_is_conc[-3:].sum()) if wnd_is_conc.size >= 3 else float(wnd_is_conc.sum())
                seq_feats.append([total_conc, avg_dt, last3])
                groups.append(rid)

    # Keep windows as list (avoid ragged object arrays)
    y = np.asarray(y_list, dtype=np.int32)
    seq_feats = np.asarray(seq_feats, dtype=np.float32)
    groups = np.asarray(groups)
    return X_seqs, y, seq_feats, groups


def pad_and_scale_sequences(X_train_list, X_test_list, seq_train_feats, seq_test_feats):
    max_len = max(max(len(s) for s in X_train_list), max(len(s) for s in X_test_list))
    # Pad to equal length
    X_train = pad_sequences(X_train_list, maxlen=max_len, dtype="float32", padding="post", truncating="post")
    X_test  = pad_sequences(X_test_list,  maxlen=max_len, dtype="float32", padding="post", truncating="post")

    # Scale per feature across flattened timesteps (then reshape)
    scaler_seq = StandardScaler()
    X_train_scaled = scaler_seq.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled  = scaler_seq.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Scale sequence-level features
    scaler_seq_level = StandardScaler()
    seq_train_scaled = scaler_seq_level.fit_transform(seq_train_feats)
    seq_test_scaled  = scaler_seq_level.transform(seq_test_feats)

    return X_train_scaled, X_test_scaled, seq_train_scaled, seq_test_scaled


def build_lstm_model(input_timesteps, input_features, seq_level_dim, lr=2.5e-4):
    inp_seq = tf.keras.Input(shape=(input_timesteps, input_features))
    x = layers.Masking(mask_value=0.0)(inp_seq)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.4)(x)

    inp_seq_level = tf.keras.Input(shape=(seq_level_dim,))
    y = layers.Dense(32, activation="relu")(inp_seq_level)

    merged = layers.concatenate([x, y])
    out = layers.Dense(1, activation="sigmoid")(merged)

    model = tf.keras.Model(inputs=[inp_seq, inp_seq_level], outputs=out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", f1_metric])
    return model


def window_to_vector_stats(X_seqs):
    """
    Convert each [T,F] window to a fixed-length vector for RF:
      - mean over time (F)
      - std  over time (F)
      - last timestep features (F)
    Result: [3F] per window
    """
    vecs = []
    for w in X_seqs:
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        mean = np.nanmean(w, axis=0)
        std  = np.nanstd(w, axis=0)
        last = w[-1]
        vecs.append(np.concatenate([mean, std, last], axis=0).astype(np.float32))
    return np.asarray(vecs, dtype=np.float32)


def compute_class_weights(y):
    classes = np.unique(y)
    if classes.size == 1:
        return None
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(ww) for c, ww in zip(classes, w)}


def best_threshold_from_pr(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
    # thresholds length = len(prec) - 1, align with f1s[1:]
    if len(f1s) <= 1 or len(thr) == 0:
        return 0.5
    best_idx = np.nanargmax(f1s[1:]) + 1
    return float(thr[best_idx - 1])


def plot_roc(y_true, y_score, label, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()


def plot_pr(y_true, y_score, label, out_png):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6,4))
    plt.plot(rec, prec, label=label)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve'); plt.legend(loc='lower left'); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()


def plot_cm(y_true, y_pred, label, out_png):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Confusion Matrix: {label}')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['No-Conflict','Conflict'])
    plt.yticks(ticks, ['No-Conflict','Conflict'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()


# =========================
# Main
# =========================
def main():
    args = parse_args()

    # Load & sanity
    df = pd.read_csv(args.csv)
    ensure_columns(df, [
        'resource_id','attribute','user_role','modification_time',
        'is_concurrent','criticality_score','attribute_sensitivity_score',
        'change_magnitude','conflict_after_sequence'
    ])

    # Build engineered event-level features
    df = build_basic_features(df)

    # Build sliding-window sequences
    X_seqs_list, y_all, seq_feats_all, groups_all = window_sequences(
        df, window=args.window, stride=args.stride
    )

    # Quick stats
    print(f"\nWindows built: {len(X_seqs_list)}  (window={args.window}, stride={args.stride})")
    print("Label distribution (window level):")
    print(pd.Series(y_all).value_counts())

    # Convert for RF (window-level vectors)
    X_rf_all = window_to_vector_stats(X_seqs_list)

    # CV setup (stratified by y, grouped by resource to avoid leakage)
    folds = max(2, args.folds)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)

    # Collect metrics per fold
    metrics_cols = ["Precision","Recall","F1","AUC","MCC","Kappa"]
    rf_scores = []
    lstm_scores = []

    fold_idx = 0
    for train_idx, test_idx in cv.split(X_rf_all, y_all, groups=groups_all):
        fold_idx += 1
        print(f"\n=== Fold {fold_idx}/{folds} ===")

        # Split sets
        X_rf_tr, X_rf_te = X_rf_all[train_idx], X_rf_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        groups_tr, groups_te = groups_all[train_idx], groups_all[test_idx]

        # For LSTM: need sequence tensors & seq-level features per split
        X_seq_tr_list = [X_seqs_list[i] for i in train_idx]
        X_seq_te_list = [X_seqs_list[i] for i in test_idx]
        seq_feat_tr = seq_feats_all[train_idx]
        seq_feat_te = seq_feats_all[test_idx]

        # LSTM train/val split within fold (group-aware)
        uniq_groups_tr = np.unique(groups_tr)
        g_tr, g_val = train_test_split(uniq_groups_tr, test_size=args.val_frac, random_state=42, stratify=None)
        tr_mask = np.isin(groups_tr, g_tr)
        val_mask = np.isin(groups_tr, g_val)

        # Prepare padded & scaled tensors
        X_seq_tr_scaled, X_seq_te_scaled, seq_tr_scaled, seq_te_scaled = pad_and_scale_sequences(
            X_seq_tr_list, X_seq_te_list, seq_feat_tr, seq_feat_te
        )

        # Build train/val tensors from the *train split*
        X_seq_train = X_seq_tr_scaled[tr_mask]
        X_seq_val   = X_seq_tr_scaled[val_mask]
        y_train     = y_tr[tr_mask]
        y_val       = y_tr[val_mask]
        seq_train   = seq_tr_scaled[tr_mask]
        seq_val     = seq_tr_scaled[val_mask]

        # ---------------- RF (window-level) ----------------
        rf = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1
        )
        rf.fit(X_rf_tr, y_tr)
        rf_prob = rf.predict_proba(X_rf_te)[:,1]
        thr_rf = 0.5  # option: tune via PR on train
        rf_pred = (rf_prob >= thr_rf).astype(int)

        rf_prec = precision_score(y_te, rf_pred, zero_division=0)
        rf_rec  = recall_score(y_te, rf_pred, zero_division=0)
        rf_f1   = f1_score(y_te, rf_pred, zero_division=0)
        rf_auc  = roc_auc_score(y_te, rf_prob)
        rf_mcc  = matthews_corrcoef(y_te, rf_pred)
        rf_kap  = cohen_kappa_score(y_te, rf_pred)
        rf_scores.append([rf_prec, rf_rec, rf_f1, rf_auc, rf_mcc, rf_kap])

        print(f"RF:   P={rf_prec:.4f} R={rf_rec:.4f} F1={rf_f1:.4f} AUC={rf_auc:.4f} MCC={rf_mcc:.4f} Kappa={rf_kap:.4f}")

        # ---------------- LSTM (sequence-level) ----------------
        class_w = compute_class_weights(y_train)

        model = build_lstm_model(
            input_timesteps=X_seq_train.shape[1],
            input_features=X_seq_train.shape[2],
            seq_level_dim=seq_train.shape[1],
            lr=2.5e-4
        )

        es = callbacks.EarlyStopping(monitor="val_f1_metric", mode="max", patience=15, restore_best_weights=True, verbose=1)
        rl = callbacks.ReduceLROnPlateau(monitor="val_f1_metric", mode="max", factor=0.5, patience=7, min_lr=1e-7, verbose=1)

        history = model.fit(
            [X_seq_train, seq_train], y_train,
            validation_data=([X_seq_val, seq_val], y_val),
            epochs=args.epochs, batch_size=args.batch,
            callbacks=[es, rl], class_weight=class_w, verbose=1
        )

        # Choose threshold on validation PR curve
        val_prob = model.predict([X_seq_val, seq_val]).ravel()
        thr = best_threshold_from_pr(y_val, val_prob)

        # Test
        lstm_prob = model.predict([X_seq_te_scaled, seq_te_scaled]).ravel()
        lstm_pred = (lstm_prob >= thr).astype(int)

        lstm_prec = precision_score(y_te, lstm_pred, zero_division=0)
        lstm_rec  = recall_score(y_te, lstm_pred, zero_division=0)
        lstm_f1   = f1_score(y_te, lstm_pred, zero_division=0)
        lstm_auc  = roc_auc_score(y_te, lstm_prob)
        lstm_mcc  = matthews_corrcoef(y_te, lstm_pred)
        lstm_kap  = cohen_kappa_score(y_te, lstm_pred)
        lstm_scores.append([lstm_prec, lstm_rec, lstm_f1, lstm_auc, lstm_mcc, lstm_kap])

        print(f"LSTM: P={lstm_prec:.4f} R={lstm_rec:.4f} F1={lstm_f1:.4f} AUC={lstm_auc:.4f} MCC={lstm_mcc:.4f} Kappa={lstm_kap:.4f}")

        # Save plots for last fold only to keep it light
        if fold_idx == folds:
            prefix = args.plots_prefix
            plot_roc(y_te, rf_prob, "Random Forest (window-agg)", f"{prefix}_rf_roc.png")
            plot_pr(y_te, rf_prob, "Random Forest (window-agg)", f"{prefix}_rf_pr.png")
            plot_cm(y_te, rf_pred, "Random Forest (window-agg)", f"{prefix}_rf_cm.png")

            plot_roc(y_te, lstm_prob, "LSTM Combined", f"{prefix}_lstm_roc.png")
            plot_pr(y_te, lstm_prob, "LSTM Combined", f"{prefix}_lstm_pr.png")
            plot_cm(y_te, lstm_pred, "LSTM Combined", f"{prefix}_lstm_cm.png")
            print(f"\nSaved plots: {prefix}_rf_roc.png, {prefix}_rf_pr.png, {prefix}_rf_cm.png, "
                  f"{prefix}_lstm_roc.png, {prefix}_lstm_pr.png, {prefix}_lstm_cm.png")

    # Summaries
    rf_scores = np.asarray(rf_scores, dtype=float)
    lstm_scores = np.asarray(lstm_scores, dtype=float)

    def mean_std(arr):
        return arr.mean(axis=0), arr.std(axis=0)

    headers = ["Precision","Recall","F1","AUC","MCC","Kappa"]
    rf_mu, rf_sd = mean_std(rf_scores)
    lstm_mu, lstm_sd = mean_std(lstm_scores)

    print("\n=== Cross-Validation Summary (mean ± std) ===")
    print("Random Forest (window-agg):")
    for h, m, s in zip(headers, rf_mu, rf_sd):
        print(f"  {h}: {m:.4f} ± {s:.4f}")
    print("LSTM Combined:")
    for h, m, s in zip(headers, lstm_mu, lstm_sd):
        print(f"  {h}: {m:.4f} ± {s:.4f}")

    # Export a small CSV summary
    out = pd.DataFrame({
        "Metric": headers,
        "RF_mean": rf_mu, "RF_std": rf_sd,
        "LSTM_mean": lstm_mu, "LSTM_std": lstm_sd
    })
    out.to_csv("cv_summary_metrics.csv", index=False)
    print("\nSaved: cv_summary_metrics.csv")


if __name__ == "__main__":
    main()
