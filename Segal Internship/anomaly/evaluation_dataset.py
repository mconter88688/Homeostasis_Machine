import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import load_model
from constants import UCSD_TEST_PATH, UCSD_GT_PATH, BEST_MODEL_PATH, SEQ_LEN, INPUT_SHAPE, ANOMALY_THRESHOLD

def load_test_sequences():
    X = []
    y_true = []

    for vid in sorted(os.listdir(UCSD_TEST_PATH)):
        vid_path = os.path.join(UCSD_TEST_PATH, vid)
        if "_gt" in vid or not os.path.isdir(vid_path):
            continue

        gt_path = vid_path + "_gt"
        frames = sorted(os.listdir(vid_path))
        gt_frames = sorted(os.listdir(gt_path))

        for i in range(len(frames) - SEQ_LEN + 1):
            seq = []
            is_anomaly = 0
            for j in range(i, i + SEQ_LEN):
                # Load frame
                img_path = os.path.join(vid_path, frames[j])
                img = cv2.imread(img_path)
                img = cv2.resize(img, INPUT_SHAPE[:2])
                img = img.astype("float32") / 255.0
                seq.append(img)

                # Load GT mask
                gt_img_path = os.path.join(gt_path, gt_frames[j])
                gt_mask = cv2.imread(gt_img_path, 0)  # grayscale
                if gt_mask is not None and np.max(gt_mask) > 0:
                    is_anomaly = 1

            if len(seq) == SEQ_LEN:
                X.append(np.array(seq))
                y_true.append(is_anomaly)

    return np.array(X), np.array(y_true)


if __name__ == "__main__":
    print("Loading test sequences...")
    X_test, y_true = load_test_sequences()
    print(f"Loaded {len(y_true)} test sequences")

    print("Loading model...")
    model = load_model(BEST_MODEL_PATH)

    print("Evaluating...")
    y_prob = model.predict(X_test, verbose=0).squeeze()
    y_pred = (y_prob > ANOMALY_THRESHOLD).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"Test Accuracy: {acc:.3f}, AUC: {auc:.3f}")
