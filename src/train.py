# train.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score
)

from args import parse_args
from utils import set_seeds, ensure_outdir, save_json
from features import ensure_columns, build_basic_features, window_sequences, pad_and_scale_sequences, window_to_vector_stats
from models import build_lstm_model, build_rf, compute_class_weights
from metrics import best_threshold_from_pr, plot_roc, plot_pr, plot_cm

def run_fold(
    fold_idx, folds, *,
    X_rf_all, X_seqs_list, y_all, seq_feats_all, groups_all,
    args
):
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    all_splits = list(cv.split(X_rf_all, y_all, groups=groups_all))
    train_idx, test_idx = all_splits[fold_idx-1]

    X_rf_tr, X_rf_te = X_rf_all[train_idx], X_rf_all[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    groups_tr, groups_te = groups_all[train_idx], groups_all[test_idx]

    X_seq_tr_list = [X_seqs_list[i] for i in train_idx]
    X_seq_te_list = [X_seqs_list[i] for i in test_idx]
    seq_feat_tr = seq_feats_all[train_idx]
    seq_feat_te = seq_feats_all[test_idx]

    uniq_groups_tr = np.unique(groups_tr)
    g_tr, g_val = train_test_split(
        uniq_groups_tr, test_size=args.val_frac, random_state=args.seed, stratify=None
    )
    tr_mask = np.isin(groups_tr, g_tr)
    val_mask = np.isin(groups_tr, g_val)

    X_seq_tr_scaled, X_seq_te_scaled, seq_tr_scaled, seq_te_scaled = pad_and_scale_sequences(
        X_seq_tr_list, X_seq_te_list, seq_feat_tr, seq_feat_te
    )
    X_seq_train = X_seq_tr_scaled[tr_mask]
    X_seq_val   = X_seq_tr_scaled[val_mask]
    y_train     = y_tr[tr_mask]
    y_val       = y_tr[val_mask]
    seq_train   = seq_tr_scaled[tr_mask]
    seq_val     = seq_tr_scaled[val_mask]

    # --- RF ---
    rf = build_rf(random_state=args.seed)
    rf.fit(X_rf_tr, y_tr)
    rf_prob = rf.predict_proba(X_rf_te)[:,1]
    rf_pred = (rf_prob >= 0.5).astype(int)

    rf_metrics = {
        "Precision": precision_score(y_te, rf_pred, zero_division=0),
        "Recall":    recall_score(y_te, rf_pred, zero_division=0),
        "F1":        f1_score(y_te, rf_pred, zero_division=0),
        "AUC":       roc_auc_score(y_te, rf_prob),
        "MCC":       matthews_corrcoef(y_te, rf_pred),
        "Kappa":     cohen_kappa_score(y_te, rf_pred),
    }

    # --- LSTM ---
    class_w = compute_class_weights(y_train)
    model = build_lstm_model(
        input_timesteps=X_seq_train.shape[1],
        input_features=X_seq_train.shape[2],
        seq_level_dim=seq_train.shape[1],
        lr=2.5e-4
    )
    from tensorflow.keras import callbacks
    es = callbacks.EarlyStopping(
        monitor="val_f1_metric", mode="max", patience=15,
        restore_best_weights=True, verbose=1
    )
    rl = callbacks.ReduceLROnPlateau(
        monitor="val_f1_metric", mode="max", factor=0.5,
        patience=7, min_lr=1e-7, verbose=1
    )
    model.fit(
        [X_seq_train, seq_train], y_train,
        validation_data=([X_seq_val, seq_val], y_val),
        epochs=args.epochs, batch_size=args.batch,
        callbacks=[es, rl], class_weight=class_w, verbose=1
    )

    val_prob = model.predict([X_seq_val, seq_val]).ravel()
    thr = best_threshold_from_pr(y_val, val_prob)

    lstm_prob = model.predict([X_seq_te_scaled, seq_te_scaled]).ravel()
    lstm_pred = (lstm_prob >= thr).astype(int)

    lstm_metrics = {
        "Precision": precision_score(y_te, lstm_pred, zero_division=0),
        "Recall":    recall_score(y_te, lstm_pred, zero_division=0),
        "F1":        f1_score(y_te, lstm_pred, zero_division=0),
        "AUC":       roc_auc_score(y_te, lstm_prob),
        "MCC":       matthews_corrcoef(y_te, lstm_pred),
        "Kappa":     cohen_kappa_score(y_te, lstm_pred),
        "Threshold": float(thr),
    }

    return rf_metrics, lstm_metrics, (rf_prob, rf_pred, lstm_prob, lstm_pred, y_te)

def main():
    args = parse_args()
    set_seeds(args.seed)
    ensure_outdir(args.outdir)

    df = pd.read_csv(args.csv)
    ensure_columns(df, [
        'resource_id','attribute','user_role','modification_time',
        'is_concurrent','criticality_score','attribute_sensitivity_score',
        'change_magnitude','conflict_after_sequence'
    ])

    df = build_basic_features(df)

    X_seqs_list, y_all, seq_feats_all, groups_all = window_sequences(
        df, window=args.window, stride=args.stride
    )
    print(f"\nWindows built: {len(X_seqs_list)}  (window={args.window}, stride={args.stride})")
    print("Label distribution (window level):")
    print(pd.Series(y_all).value_counts())

    X_rf_all = window_to_vector_stats(X_seqs_list)

    folds = max(2, args.folds)
    headers = ["Precision","Recall","F1","AUC","MCC","Kappa"]
    rf_scores, lstm_scores = [], []
    last_artifacts = None

    for fold_idx in range(1, folds+1):
        print(f"\n=== Fold {fold_idx}/{folds} ===")
        rf_m, lstm_m, artifacts = run_fold(
            fold_idx, folds,
            X_rf_all=X_rf_all, X_seqs_list=X_seqs_list, y_all=y_all,
            seq_feats_all=seq_feats_all, groups_all=groups_all,
            args=args
        )
        rf_scores.append([rf_m[h] for h in headers])
        lstm_scores.append([lstm_m[h] for h in headers])

        # Save per-fold metrics
        fold_df = pd.DataFrame({
            "Metric": headers,
            "RF": [rf_m[h] for h in headers],
            "LSTM": [lstm_m[h] for h in headers],
        })
        fold_df.to_csv(os.path.join(args.outdir, f"fold_{fold_idx}_metrics.csv"), index=False)

        print("RF  :", " ".join([f"{k}={rf_m[k]:.4f}" for k in headers]))
        print("LSTM:", " ".join([f"{k}={lstm_m[k]:.4f}" for k in headers]), f"Thr={lstm_m['Threshold']:.4f}")

        if fold_idx == folds:
            last_artifacts = (artifacts, rf_m, lstm_m)

    rf_scores = np.asarray(rf_scores, dtype=float)
    lstm_scores = np.asarray(lstm_scores, dtype=float)
    rf_mu, rf_sd = rf_scores.mean(axis=0), rf_scores.std(axis=0)
    lstm_mu, lstm_sd = lstm_scores.mean(axis=0), lstm_scores.std(axis=0)

    print("\n=== Cross-Validation Summary (mean ± std) ===")
    print("Random Forest (window-agg):")
    for h, m, s in zip(headers, rf_mu, rf_sd):
        print(f"  {h}: {m:.4f} ± {s:.4f}")
    print("LSTM Combined:")
    for h, m, s in zip(headers, lstm_mu, lstm_sd):
        print(f"  {h}: {m:.4f} ± {s:.4f}")

    summary_df = pd.DataFrame({
        "Metric": headers,
        "RF_mean": rf_mu, "RF_std": rf_sd,
        "LSTM_mean": lstm_mu, "LSTM_std": lstm_sd
    })
    summary_df.to_csv(os.path.join(args.outdir, "cv_summary_metrics.csv"), index=False)

    # plots for last fold
    if last_artifacts is not None:
        (artifacts, rf_m_last, lstm_m_last) = last_artifacts
        (rf_prob, rf_pred, lstm_prob, lstm_pred, y_te) = artifacts
        prefix = os.path.join(args.outdir, args.plots_prefix)

        plot_roc(y_te, rf_prob, "Random Forest (window-agg)", f"{prefix}_rf_roc.png")
        plot_pr(y_te, rf_prob, "Random Forest (window-agg)", f"{prefix}_rf_pr.png")
        plot_cm(y_te, rf_pred, "Random Forest (window-agg)", f"{prefix}_rf_cm.png")

        plot_roc(y_te, lstm_prob, "LSTM Combined", f"{prefix}_lstm_roc.png")
        plot_pr(y_te, lstm_prob, "LSTM Combined", f"{prefix}_lstm_pr.png")
        plot_cm(y_te, lstm_pred, "LSTM Combined", f"{prefix}_lstm_cm.png")

        save_json(
            {"RandomForest": rf_m_last, "LSTM": lstm_m_last},
            os.path.join(args.outdir, "last_fold_metrics.json")
        )

if __name__ == "__main__":
    main()
