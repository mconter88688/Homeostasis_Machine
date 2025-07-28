import numpy as np
import os, glob, cv2
from scipy.io import loadmat
import tensorflow as tf
import constants as cons
import tensorflow_hub as hub

feature_extractor = hub.KerasLayer(cons.EFFICIENT_NET_B0, 
                                   input_shape=cons.INPUT_SHAPE, 
                                   trainable=False)

def extract_feature(frame):
    resized = cv2.resize(frame, cons.INPUT_SHAPE[:2]) / 255.0
    tensor = tf.expand_dims(resized.astype(np.float32), axis=0)
    return tf.squeeze(feature_extractor(tensor)).numpy()

def load_gt_labels(mat_file, total_frames):
    mat = loadmat(mat_file)
    anomaly_ranges = mat['gt'][0]
    labels = np.zeros(total_frames, dtype=int)
    for r in anomaly_ranges:
        start, end = r[0]-1, r[1]   # Convert to 0-based index
        labels[start:end] = 1
    return labels

def load_data_sequence(frame_folder, gt_file):
    frames = sorted(glob.glob(os.path.join(frame_folder, "*.png")))
    labels = load_gt_labels(gt_file, len(frames))

    feats, X, y = [], [], []
    for f in frames:
        img = cv2.imread(f)
        if img is None: continue
        feats.append(extract_feature(img))

    for i in range(len(feats) - cons.SEQ_LEN + 1):
        X.append(np.stack(feats[i:i+cons.SEQ_LEN]))
        y.append(1 if any(labels[i:i+cons.SEQ_LEN]) else 0)
    
    return np.array(X), np.array(y)
