# make_synthetic_iac.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# =========================
# Config
# =========================
NUM_SEQUENCES = 10_000
MAX_SEQUENCE_LENGTH = 20

# Temporal dynamics
CONCURRENCY_WINDOW_SECONDS = 5           # window to consider concurrent edits
EXP_MEAN_GAP_SECONDS = 600               # mean of exponential gap between edits (bursty sequences)
HORIZON_SECONDS = 300                    # for "conflict_within_h" (label lookahead horizon)

# Resource universe
NUM_RESOURCES = 75
ATTRIBUTES_PER_RESOURCE = 7
USER_ROLES = ["developer", "operator", "admin"]

# Dependency graph (DAG-ish, sparse)
AVG_DEPENDENCIES = 2                     # average out-degree
DEPENDENCY_PROB = min(1.0, AVG_DEPENDENCIES / max(1, NUM_RESOURCES - 1))

# Desired class balance (event-level)
TARGET_POS_FRAC = 0.50                   # try 0.4–0.6 as needed

# Random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# =========================
# Static maps
# =========================
resource_ids = [f"resource_{i}" for i in range(NUM_RESOURCES)]
attributes = [f"attr_{i}" for i in range(ATTRIBUTES_PER_RESOURCE)]

resource_criticality = {rid: np.random.uniform(0.1, 0.9) for rid in resource_ids}
attribute_sensitivity = {att: np.random.uniform(0.1, 0.9) for att in attributes}
role_risk = {"developer": 0.0, "operator": 0.15, "admin": 0.25}  # small role prior

# =========================
# Dependency Graph
# =========================
# Simple random directed graph (avoid self-deps). Intended to be sparse.
depends_on = {rid: [] for rid in resource_ids}
for src in resource_ids:
    for dst in resource_ids:
        if src == dst:
            continue
        # Make graph sparser; avoid too many edges
        if np.random.rand() < DEPENDENCY_PROB / NUM_RESOURCES:
            depends_on[src].append(dst)

# Precompute degree / centrality-like features
dep_degree = {rid: len(depends_on[rid]) for rid in resource_ids}

# =========================
# Helpers
# =========================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def exp_time_jump():
    """Exponential inter-arrival gap (bursty edits)."""
    gap = np.random.exponential(scale=EXP_MEAN_GAP_SECONDS)
    # clamp to a reasonable max to avoid extreme outliers
    return int(max(1, min(gap, 7200)))

def generate_sequence():
    """Generate one sequence of edits; possibly switching resources."""
    rid = random.choice(resource_ids)
    seq = []
    # start somewhere in the last ~150 hours
    current_time = datetime.now() - timedelta(hours=random.randint(1, 150))

    recent_by_res = {}  # resource_id -> last time seen (for concurrency detection)
    for _ in range(random.randint(5, MAX_SEQUENCE_LENGTH)):
        attribute = random.choice(attributes)
        change_frequency = np.random.uniform(0.01, 0.8)
        change_magnitude = np.random.uniform(0.01, 1.0)
        user_role = random.choice(USER_ROLES)

        # exponential gap → bursty sequences
        current_time += timedelta(seconds=exp_time_jump())

        # concurrency: close in time to ANY tracked resource
        is_concurrent = False
        related_resource_id = None
        for other_res, t in list(recent_by_res.items()):
            if (current_time - t).total_seconds() < CONCURRENCY_WINDOW_SECONDS:
                is_concurrent = True
                related_resource_id = other_res
                break

        # dependency features (count recent modifications among dependencies)
        dep_list = depends_on[rid]
        dep_recent_count = 0
        for d in dep_list:
            t = recent_by_res.get(d, None)
            if t is not None and (current_time - t).total_seconds() < CONCURRENCY_WINDOW_SECONDS:
                dep_recent_count += 1
        dep_recent_flag = 1 if dep_recent_count > 0 else 0

        seq.append({
            "resource_id": rid,
            "attribute": attribute,
            "change_frequency": change_frequency,
            "modification_time": current_time,
            "is_concurrent": is_concurrent,
            "related_resource_id": related_resource_id if is_concurrent else None,
            "criticality_score": resource_criticality[rid],
            "attribute_sensitivity_score": attribute_sensitivity[attribute],
            "change_magnitude": change_magnitude,
            "user_role": user_role,
            "dep_degree": dep_degree[rid],
            "dep_recent_count": dep_recent_count,
            "dep_recent_flag": dep_recent_flag,
        })

        # update recency map
        recent_by_res[rid] = current_time

        # occasionally switch resources to induce cross-resource concurrency windows
        if random.random() < 0.15:
            rid = random.choice(resource_ids)
            # prune stale entries
            cutoff = current_time - timedelta(seconds=CONCURRENCY_WINDOW_SECONDS * 2)
            recent_by_res = {k: v for k, v in recent_by_res.items() if v > cutoff}

    return pd.DataFrame(seq)

# =========================
# Generate raw sequences → flat events table
# =========================
sequences = [generate_sequence() for _ in range(NUM_SEQUENCES)]
events = pd.concat(sequences, ignore_index=True)

# =========================
# Derived per-event features
# =========================
events = events.sort_values(["resource_id", "modification_time"]).reset_index(drop=True)

# time since last change per resource
events["modification_time"] = pd.to_datetime(events["modification_time"])
events["mod_time_sec"] = events["modification_time"].view("int64") / 1e9
events["time_since_last_change"] = (
    events.groupby("resource_id")["mod_time_sec"].diff().fillna(0.0)
)

# short-term concurrency count (rolling window size 3)
events["concurrent_count_3"] = (
    events.groupby("resource_id")["is_concurrent"]
          .transform(lambda s: s.astype(float).rolling(3, min_periods=1).sum())
          .fillna(0.0)
)

# local frequency momentum
events["change_frequency_diff"] = (
    events.groupby("resource_id")["change_frequency"].diff().fillna(0.0)
)

# role risk numeric
events["role_risk"] = events["user_role"].map(role_risk).astype(float)

# =========================
# Per-event conflict probability (logistic model with deps)
# =========================
w = {
    "bias": -0.6,           # global intercept
    "is_concurrent": 1.2,
    "criticality": 1.0,
    "sensitivity": 1.1,
    "magnitude": 1.3,
    "freq": 0.6,
    "freq_diff": 0.3,
    "recent": -0.25,        # longer since last change → safer
    "concurrent_cnt": 0.4,
    "role": 0.5,
    "dep_recent_flag": 0.9, # any dependency changed very recently
    "dep_recent_cnt": 0.25, # number of deps changed recently
    "dep_degree": 0.08,     # more dependencies → more coupling risk
}

recent_scaled = np.tanh(events["time_since_last_change"].values / 600.0)  # saturate
logit = (
    w["bias"]
    + w["is_concurrent"]   * events["is_concurrent"].astype(float).values
    + w["criticality"]     * events["criticality_score"].values
    + w["sensitivity"]     * events["attribute_sensitivity_score"].values
    + w["magnitude"]       * events["change_magnitude"].values
    + w["freq"]            * events["change_frequency"].values
    + w["freq_diff"]       * events["change_frequency_diff"].values
    + w["recent"]          * recent_scaled
    + w["concurrent_cnt"]  * events["concurrent_count_3"].values
    + w["role"]            * events["role_risk"].values
    + w["dep_recent_flag"] * events["dep_recent_flag"].astype(float).values
    + w["dep_recent_cnt"]  * events["dep_recent_count"].astype(float).values
    + w["dep_degree"]      * events["dep_degree"].astype(float).values
)
p_raw = sigmoid(logit)

# small label noise for realism
noise = np.random.uniform(-0.02, 0.02, size=len(p_raw))
p = np.clip(p_raw + noise, 0.0, 1.0)

# Balance: choose τ so that ~TARGET_POS_FRAC events are positive
tau = float(np.quantile(p, 1.0 - TARGET_POS_FRAC))
y = (p >= tau).astype(int)

events["conflict_prob"] = p
events["conflict_after_event"] = y  # per-event "hard" label

# =========================
# Horizon label: conflict within next H seconds (same resource)
# =========================
H = HORIZON_SECONDS
events["conflict_within_h"] = 0
for rid, g in events.groupby("resource_id", sort=False):
    idx = g.index.values
    t = g["mod_time_sec"].values
    y_future = g["conflict_after_event"].values

    # For each event i, mark 1 if any future event within H seconds is a conflict
    # Efficient two-pointer scan
    j = 0
    for i in range(len(idx)):
        # advance j to be > i
        j = max(j, i + 1)
        # move j while within horizon
        while j < len(idx) and (t[j] - t[i]) <= H:
            if y_future[j] == 1:
                events.loc[idx[i], "conflict_within_h"] = 1
                break
            j += 1

# =========================
# Save event-level dataset
# (keep column name expected by your training script: 'conflict_after_sequence')
# =========================
out = events[[
    "resource_id","attribute","user_role","modification_time","is_concurrent",
    "criticality_score","attribute_sensitivity_score","change_magnitude",
    "change_frequency","time_since_last_change","concurrent_count_3","change_frequency_diff",
    "dep_degree","dep_recent_count","dep_recent_flag",
    "role_risk","conflict_prob","conflict_after_event","conflict_within_h"
]].copy()

# Backward-compatible column name for training
out = out.rename(columns={"conflict_after_event": "conflict_after_sequence"})

out.to_csv("synthetic_sequential_iac_data.csv", index=False)

pos_rate = out["conflict_after_sequence"].mean()
print("Saved synthetic_sequential_iac_data.csv")
print(f"Approx positive rate: {pos_rate:.3f} (target {TARGET_POS_FRAC:.2f})")
print(f"Rows: {len(out):,} | Resources: {NUM_RESOURCES} | Avg deps: {np.mean([len(depends_on[r]) for r in resource_ids]):.2f}")
