import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import constants as cons
from models import build_one_way_7_18

# Load model and feature extractor
model = load_model(cons.MODEL_PATH)
feature_extractor = hub.KerasLayer(cons.EFFICIENT_NET_B0, input_shape=cons.INPUT_SHAPE, trainable=False)

def extract_feature(frame):
    resized = cv2.resize(frame, cons.INPUT_SHAPE[:2]) / 255.0
    tensor = tf.expand_dims(resized.astype(np.float32), axis=0)
    return tf.squeeze(feature_extractor(tensor)).numpy()

def load_data_sequence(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    labels = []
    features = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        feat = extract_feature(img)
        features.append(feat)

        # Labeling rule — change this depending on how you name anomalous frames
        label = 1 if "anomaly" in os.path.basename(path).lower() else 0
        labels.append(label)
    
    X, y = [], []
    for i in range(len(features) - cons.SEQ_LEN + 1):
        X.append(np.stack(features[i:i+cons.SEQ_LEN]))
        y_seq = labels[i:i+cons.SEQ_LEN]
        y.append(1 if any(y_seq) else 0)
    return np.array(X), np.array(y)

def evaluate_all_sequences(base_folder):
    all_X, all_y = [], []
    folders = sorted([f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))])
    
    for folder_name in folders:
        folder_path = os.path.join(base_folder, folder_name)
        print(f"Processing: {folder_name}")
        X, y = load_data_sequence(folder_path)
        all_X.append(X)
        all_y.append(y)
    
    X_test = np.concatenate(all_X, axis=0)
    y_test = np.concatenate(all_y, axis=0)
    print(f"Total sequences: {len(y_test)}")

    y_prob = model.predict(X_test, verbose=0).squeeze()
    y_pred = (y_prob > cons.ANOMALY_THRESHOLD).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nOverall Accuracy: {acc:.3f}")
    print(f"Overall AUC: {auc:.3f}")

# Run evaluation across all folders
evaluate_all_sequences("UCSDped2/Test_frames")

