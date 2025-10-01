# models.py
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import layers, optimizers, backend as K

def f1_metric(y_true, y_pred):
    y_true = K.flatten(K.cast(y_true, "float32"))
    y_pred_bin = K.flatten(K.round(y_pred))
    tp = K.sum(y_true * y_pred_bin)
    pp = K.sum(y_pred_bin)
    ppos = K.sum(y_true)
    precision = tp / (pp + K.epsilon())
    recall = tp / (ppos + K.epsilon())
    return 2.0 * (precision * recall) / (precision + recall + K.epsilon())

def compute_class_weights(y):
    classes = np.unique(y)
    if classes.size == 1:
        return None
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(ww) for c, ww in zip(classes, w)}

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

def build_rf(random_state=42):
    return RandomForestClassifier(
        n_estimators=300, random_state=random_state, class_weight='balanced', n_jobs=-1
    )
