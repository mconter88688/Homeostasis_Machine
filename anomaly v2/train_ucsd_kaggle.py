# train_ucsd_kaggle.py
import os, json, ctypes, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import constants as cons
import models as mod
from dataset_utils_kaggle import build_train_windows

# ===== Jetson: preload libs (same pattern as your main.py) =====
try:
    os.environ.setdefault(
        "LD_PRELOAD",
        "/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libatomic.so.1"
    )
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libatomic.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

# ===== GPU memory growth =====
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ===== Config knobs (env overridable) =====
EPOCHS   = int(os.getenv("EPOCHS", "20"))
BATCH    = int(os.getenv("TRAIN_BATCH", "16"))   # smaller for Jetson
VAL_PCT  = float(os.getenv("VAL_PCT", "0.1"))    # 10% val split

os.makedirs(cons.MODEL_FOLDER, exist_ok=True)
os.makedirs(cons.DATA_FOLDER, exist_ok=True)

def main():
    t0 = time.time()
    print("=== UCSD Ped2 Training (Autoencoder) ===")
    print(f"Settings -> epochs={EPOCHS}, batch={BATCH}, val_pct={VAL_PCT}")
    print("Building train windows (will cache features if needed)…")

    X = build_train_windows(seq_len=cons.SEQ_LEN, step=1, dtype=np.float32)  # (N, T, F)
    if X.shape[0] == 0:
        raise RuntimeError("No training windows found. Check dataset paths and cache.")
    y = X  # autoencoder targets

    # shuffle/split
    n = X.shape[0]
    idx = np.random.permutation(n)
    split = int((1.0 - VAL_PCT) * n)
    tr, va = idx[:split], idx[split:]
    Xtr, Xva = X[tr], X[va]

    print(f"Train windows: {Xtr.shape[0]}  |  Val windows: {Xva.shape[0]}")
    print(f"Window shape: (T={cons.SEQ_LEN}, F={cons.FEATURE_DIM})")

    model = mod.build_model()
    model.summary()

    # Callbacks with clear “best model saved” logs
    cbs = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(cons.BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    print("\n--- Starting training (progress below) ---")
    hist = model.fit(
        Xtr, Xtr,
        validation_data=(Xva, Xva),
        epochs=EPOCHS,
        batch_size=BATCH,
        shuffle=True,
        callbacks=cbs,
        verbose=1  # progress bar like "Epoch 2/20 …"
    )

    print("\nSaving final model…")
    model.save(cons.MODEL_PATH)
    print(f"Saved model to: {cons.MODEL_PATH}")

    print("Computing validation reconstruction errors & threshold (95th pct)…")
    # Show progress on prediction as well
    recon_va = model.predict(Xva, verbose=1)  # progress bar
    va_err = np.mean((recon_va - Xva) ** 2, axis=(1,2))
    thr = float(np.percentile(va_err, 95))

    thr_path = os.path.join(cons.DATA_FOLDER, "ucsd_ped2_threshold.json")
    with open(thr_path, "w") as f:
        json.dump({"threshold": thr}, f)
    print(f"Saved threshold = {thr:.6f}  ->  {thr_path}")

    dt = time.time() - t0
    print(f"\n=== Training complete in {dt/60:.1f} min ===")

if __name__ == "__main__":
    main()
