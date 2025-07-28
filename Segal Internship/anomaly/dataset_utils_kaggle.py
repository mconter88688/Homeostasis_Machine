import os, cv2
import numpy as np
import constants as cons
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

# Load EfficientNet feature extractor
feature_extractor = hub.KerasLayer(
    cons.EFFICIENT_NET_B0,
    input_shape=cons.INPUT_SHAPE,
    trainable=False
)

def load_ucsd_sequences(seq_len=cons.SEQ_LEN):
    X, y = [], []
    tests = sorted([d for d in os.listdir(cons.UCSD_FRAMES_PATH) if d.startswith("Test") and not d.endswith("_gt")])

    for test in tests:
        frame_dir = os.path.join(cons.UCSD_FRAMES_PATH, test)
        gt_dir = os.path.join(cons.UCSD_FRAMES_PATH, test + "_gt")

        frames = sorted(os.listdir(frame_dir))
        gts = sorted(os.listdir(gt_dir))

        buffer_frames, buffer_labels = [], []
        for f, g in zip(frames, gts):
            frame_path = os.path.join(frame_dir, f)
            gt_path = os.path.join(gt_dir, g)

            img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (cons.INPUT_SHAPE[1], cons.INPUT_SHAPE[0])) / 255.0
            tensor = tf.expand_dims(img.astype(np.float32), axis=0)
            feats = feature_extractor(tensor)
            feat = tf.squeeze(feats).numpy()  # (1280,)

            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            label = 1 if np.max(mask) > 0 else 0

            buffer_frames.append(feat)
            buffer_labels.append(label)

            if len(buffer_frames) == seq_len:
                X.append(np.stack(buffer_frames))
                y.append(1 if any(buffer_labels) else 0)  # anomaly if any frame anomalous
                buffer_frames.pop(0)
                buffer_labels.pop(0)

    return np.array(X), np.array(y)

def split_dataset(X, y, train_size=0.6):
    return train_test_split(X, y, train_size=train_size, random_state=42, shuffle=True)
