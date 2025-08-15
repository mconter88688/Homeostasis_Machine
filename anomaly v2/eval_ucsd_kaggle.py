# eval_ucsd_kaggle.py
import os, json, ctypes, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import constants as cons
from tensorflow.keras.models import load_model
from dataset_utils_kaggle import (
    build_test_features,
    make_windows,
    window_scores_to_frame_scores,
    smooth_scores,
    load_ped2_frame_gt,
)

# ===== Jetson preload =====
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

def load_threshold(default_thr: float) -> float:
    path = os.path.join(cons.DATA_FOLDER, "ucsd_ped2_threshold.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return float(json.load(f).get("threshold", default_thr))
    return default_thr

def main():
    t0 = time.time()
    print("=== UCSD Ped2 Evaluation (Autoencoder) ===")
    model_path = cons.BEST_MODEL_PATH if os.path.exists(cons.BEST_MODEL_PATH) else cons.MODEL_PATH
    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    thr = load_threshold(cons.ANOMALY_THRESHOLD)
    print(f"Using threshold = {thr:.6f}")

    print("\nExtracting (or loading cached) test features…")
    per_video = build_test_features(dtype=np.float32)  # {name: (F, 1280)}
    gt_map = load_ped2_frame_gt()

    print("\n--- Evaluating videos ---")
    all_scores, all_labels = [], []
    for name, feats in tqdm(per_video.items(), total=len(per_video), desc="Videos processed"):
        X = make_windows(feats, cons.SEQ_LEN, step=1)  # (W, T, F)
        if X.shape[0] == 0:
            tqdm.write(f"[skip] {name}: not enough frames for one window")
            continue

        # Show predict progress bar per video
        recon = model.predict(X, verbose=1)  # batch progress
        w_err = np.mean((recon - X) ** 2, axis=(1,2))  # window-level MSE

        f_scores = window_scores_to_frame_scores(w_err, feats.shape[0], cons.SEQ_LEN, step=1)
        f_scores = smooth_scores(f_scores, k=5)

        # Save raw per-video scores for inspection
        out_np = os.path.join(cons.DATA_FOLDER, f"{name}_frame_scores.npy")
        np.save(out_np, f_scores)
        tqdm.write(f"Saved scores -> {out_np}")

        if gt_map is not None and name in gt_map:
            labels = gt_map[name].astype(np.uint8)
            # normalize per-video for ROC
            s = (f_scores - f_scores.min()) / (f_scores.ptp() + 1e-8)
            all_scores.append(s)
            all_labels.append(labels)

    if all_scores:
        y_score = np.concatenate(all_scores)
        y_true  = np.concatenate(all_labels)
        auc = roc_auc_score(y_true, y_score)
        print(f"\nFrame-level ROC-AUC on Ped2 test: {auc:.4f}")
    else:
        print("\nGround truth not found; AUC not computed (scores still saved).")

    dt = time.time() - t0
    print(f"=== Evaluation complete in {dt/60:.1f} min ===")

if __name__ == "__main__":
    main()
