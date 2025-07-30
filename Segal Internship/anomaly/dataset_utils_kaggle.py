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

def load_ucsd_sequences(train=True, seq_len=cons.SEQ_LEN):
    """
    Loads UCSD Ped2 dataset sequences.
    - train=True → load from UCSD_TRAIN_PATH (all normal)
    - train=False → load from UCSD_TEST_PATH (use *_gt for anomaly labels)
    """
    X, y = [], []

    if train:
        base_path = cons.UCSD_TRAIN_PATH
        videos = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

        for vid in videos:
            vid_path = os.path.join(base_path, vid)
            frames = sorted(os.listdir(vid_path))
            if len(frames) < seq_len:
                continue

            buffer = []
            for f in frames:
                img_path = os.path.join(vid_path, f)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (cons.INPUT_SHAPE[1], cons.INPUT_SHAPE[0])) / 255.0
                tensor = tf.expand_dims(img.astype(np.float32), axis=0)
                feats = feature_extractor(tensor)
                feat = tf.squeeze(feats).numpy()
                buffer.append(feat)

                if len(buffer) == seq_len:
                    X.append(np.stack(buffer))
                    y.append(0)  # training = all normal
                    buffer.pop(0)

    else:
        base_path = cons.UCSD_TEST_PATH
        tests = sorted([d for d in os.listdir(base_path) if d.startswith("Test") and not d.endswith("_gt")])

        for test in tests:
            frame_dir = os.path.join(base_path, test)
            gt_dir = os.path.join(base_path, test + "_gt")
            frames = sorted(os.listdir(frame_dir))
            gts = sorted(os.listdir(gt_dir))

            if len(frames) < seq_len:
                continue

            buffer_frames, buffer_labels = [], []
            for f, g in zip(frames, gts):
                frame_path = os.path.join(frame_dir, f)
                gt_path = os.path.join(gt_dir, g)

                # preprocess frame
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (cons.INPUT_SHAPE[1], cons.INPUT_SHAPE[0])) / 255.0
                tensor = tf.expand_dims(img.astype(np.float32), axis=0)
                feats = feature_extractor(tensor)
                feat = tf.squeeze(feats).numpy()

                # read GT mask
                mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                label = 1 if (mask is not None and np.any(mask > 127)) else 0

                buffer_frames.append(feat)
                buffer_labels.append(label)

                if len(buffer_frames) == seq_len:
                    X.append(np.stack(buffer_frames))
                    y.append(1 if any(buffer_labels) else 0)
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
