# utils.py
import os
import json
import numpy as np
import tensorflow as tf

def set_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
