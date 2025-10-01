# features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Label encoders
    le_res = LabelEncoder()
    le_attr = LabelEncoder()
    le_user = LabelEncoder()
    df["resource_id_encoded"] = le_res.fit_transform(df["resource_id"].astype(str))
    df["attribute_encoded"]  = le_attr.fit_transform(df["attribute"].astype(str))
    df["user_role_encoded"]  = le_user.fit_transform(df["user_role"].astype(str))

    # Time to seconds
    mt = pd.to_datetime(df["modification_time"], errors="coerce")
    if mt.isna().any():
        raise ValueError("Some 'modification_time' failed to parse")
    df["modification_time_numerical"] = (mt.view("int64") / 1e9).astype("float64")

    # If 'change_frequency' missing, synthesize a simple rolling count per resource
    if "change_frequency" not in df.columns:
        df = df.sort_values(["resource_id", "modification_time"]).copy()
        df["change_frequency"] = (
            df.groupby("resource_id")["attribute"]
              .transform(lambda s: s.rolling(window=5, min_periods=1).count())
              .astype(float)
        )

    # Sort and derive sequential helpers
    df = df.sort_values(["resource_id", "modification_time"]).copy()
    df["time_since_last_change"] = (
        df.groupby("resource_id")["modification_time_numerical"].diff().fillna(0.0)
    )
    df["concurrent_count_3"] = (
        df.groupby("resource_id")["is_concurrent"].transform(
            lambda s: s.astype(float).rolling(window=3, min_periods=1).sum().fillna(0.0)
        )
    )
    df["change_frequency_diff"] = (
        df.groupby("resource_id")["change_frequency"].diff().fillna(0.0)
    )

    feat_cols = [
        "resource_id_encoded","attribute_encoded","change_frequency",
        "modification_time_numerical","is_concurrent","user_role_encoded",
        "criticality_score","attribute_sensitivity_score","change_magnitude",
        "time_since_last_change","concurrent_count_3","change_frequency_diff"
    ]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def window_sequences(df: pd.DataFrame, window=32, stride=8):
    """
    Sliding-window sequences per resource.
    Label = label at the last timestep of each window.
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

        lim = max(1, len(arr) - window + 1)
        for start in range(0, lim, stride):
            end = start + window
            if end <= len(arr):
                win = arr[start:end]
                X_seqs.append(np.asarray(win, dtype=np.float32))
                y_list.append(int(labels[end - 1]))

                wnd_is_conc = is_conc[start:end]
                wnd_times = times[start:end]
                total_conc = float(wnd_is_conc.sum())
                deltas = np.diff(wnd_times)
                avg_dt = float(deltas.mean()) if deltas.size > 0 else 0.0
                last3 = float(wnd_is_conc[-3:].sum()) if wnd_is_conc.size >= 3 else float(wnd_is_conc.sum())
                seq_feats.append([total_conc, avg_dt, last3])
                groups.append(rid)

    y = np.asarray(y_list, dtype=np.int32)
    seq_feats = np.asarray(seq_feats, dtype=np.float32)
    groups = np.asarray(groups)
    return X_seqs, y, seq_feats, groups

def pad_and_scale_sequences(X_train_list, X_test_list, seq_train_feats, seq_test_feats):
    max_len = max(max(len(s) for s in X_train_list), max(len(s) for s in X_test_list))
    X_train = pad_sequences(X_train_list, maxlen=max_len, dtype="float32", padding="post", truncating="post")
    X_test  = pad_sequences(X_test_list,  maxlen=max_len, dtype="float32", padding="post", truncating="post")

    scaler_seq = StandardScaler()
    X_train_scaled = scaler_seq.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled  = scaler_seq.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    scaler_seq_level = StandardScaler()
    seq_train_scaled = scaler_seq_level.fit_transform(seq_train_feats)
    seq_test_scaled  = scaler_seq_level.transform(seq_test_feats)

    return X_train_scaled, X_test_scaled, seq_train_scaled, seq_test_scaled

def window_to_vector_stats(X_seqs):
    """
    Convert each [T,F] window to a fixed-length vector for RF:
      - mean over time (F), std over time (F), last timestep (F) => 3F-vector.
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
