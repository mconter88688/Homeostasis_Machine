import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from constants import UCSD_TRAIN_PATH, SEQ_LEN, INPUT_SHAPE, EFFICIENT_NET_B0

# Load EfficientNet feature extractor (frozen)
feature_extractor = hub.KerasLayer(
    EFFICIENT_NET_B0,
    input_shape=INPUT_SHAPE,
    trainable=False
)

def load_ucsd_sequences(seq_len=SEQ_LEN):
    """
    Load UCSD Ped2 dataset sequences for training (normal only).
    Returns:
        X: numpy array of shape (num_sequences, seq_len, feature_dim)
        y: numpy array of shape (num_sequences,) with labels (all zeros for normal)
    """
    X = []
    y = []

    # Iterate through each training video folder
    for vid in sorted(os.listdir(UCSD_TRAIN_PATH)):
        vid_path = os.path.join(UCSD_TRAIN_PATH, vid)
        if not os.path.isdir(vid_path):
            continue

        frames = sorted(os.listdir(vid_path))
        if len(frames) < seq_len:
            continue

        buffer = []
        for f in frames:
            img_path = os.path.join(vid_path, f)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, INPUT_SHAPE[:2]) / 255.0
            tensor = tf.expand_dims(img.astype(np.float32), axis=0)
            feats = feature_extractor(tensor)
            feat = tf.squeeze(feats).numpy()  # shape: (1280,)
            buffer.append(feat)

            if len(buffer) == seq_len:
                X.append(np.stack(buffer))
                y.append(0)  # all training data = normal
                buffer.pop(0)  # slide the window

    return np.array(X), np.array(y)


def split_dataset(X, y, train_size=0.6):
    """
    Split sequences into train/test sets using NumPy only.
    """
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(train_size * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
