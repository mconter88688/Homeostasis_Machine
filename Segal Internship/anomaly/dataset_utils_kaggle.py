import os, cv2
import numpy as np
import constants as cons
import tensorflow as tf
import tensorflow_hub as hub

# Load EfficientNet feature extractor (frozen)
feature_extractor = hub.KerasLayer(
    cons.EFFICIENT_NET_B0,
    input_shape=cons.INPUT_SHAPE,
    trainable=False
)

def load_ucsd_sequences(seq_len=cons.SEQ_LEN):
    """
    Loads UCSD Ped2 dataset sequences with anomaly labels.
    A sequence is labeled anomalous if ANY frame in it has anomaly pixels.
    """
    X, y = [], []
    tests = sorted([d for d in os.listdir(cons.UCSD_FRAMES_PATH) 
                    if d.startswith("Test") and not d.endswith("_gt")])

    for test in tests:
        frame_dir = os.path.join(cons.UCSD_FRAMES_PATH, test)
        gt_dir = os.path.join(cons.UCSD_FRAMES_PATH, test + "_gt")

        frames = sorted(os.listdir(frame_dir))
        gts = sorted(os.listdir(gt_dir))

        if len(frames) < seq_len:  # skip too-short videos
            continue

        buffer_frames, buffer_labels = [], []
        for f, g in zip(frames, gts):
            frame_path = os.path.join(frame_dir, f)
            gt_path = os.path.join(gt_dir, g)

            # Read and preprocess frame
            img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (cons.INPUT_SHAPE[1], cons.INPUT_SHAPE[0])) / 255.0
            tensor = tf.expand_dims(img.astype(np.float32), axis=0)
            feats = feature_extractor(tensor)
            feat = tf.squeeze(feats).numpy()  # (1280,)

            # Read GT mask → anomaly if any white pixel
            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            label = 1 if (mask is not None and np.any(mask > 127)) else 0

            buffer_frames.append(feat)
            buffer_labels.append(label)

            # Build sequences
            if len(buffer_frames) == seq_len:
                X.append(np.stack(buffer_frames))
                y.append(1 if any(buffer_labels) else 0)  # anomaly if any frame anomalous
                buffer_frames.pop(0)
                buffer_labels.pop(0)

    return np.array(X), np.array(y)

def split_dataset(X, y, train_size=0.6):
    """
    Splits dataset into train/test without sklearn.
    """
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(train_size * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

