# args.py
import argparse

def parse_args():
    ap = argparse.ArgumentParser(
        description="IaC conflict prediction with RF (window-agg) and LSTM (sequence)"
    )
    ap.add_argument("--csv", required=True, help="Path to CSV with event-level rows")
    ap.add_argument("--window", type=int, default=32, help="Sliding window length (timesteps)")
    ap.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds (>=2)")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for LSTM")
    ap.add_argument("--epochs", type=int, default=150, help="Max epochs for LSTM")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="Fraction of train (within fold) used for validation")
    ap.add_argument("--plots_prefix", default="cv", help="Filename prefix for saved figures (last fold)")
    ap.add_argument("--outdir", default="results", help="Directory to write results and artifacts")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return ap.parse_args()
