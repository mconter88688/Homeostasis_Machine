import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from constants import UCSD_TRAIN_PATH, UCSD_TEST_PATH, SEQ_LEN, INPUT_SHAPE

def load_ucsd_sequences():
    """
    Load UCSD Ped2 dataset sequences for training (normal only).
    Returns:
        X: numpy array of shape (num_sequences, SEQ_LEN, H, W, 3)
        y: numpy array of shape (num_sequences,) with labels (all zeros for normal)
    """
    X = []
    y = []

    # iterate through each training video folder
    for vid in sorted(os.listdir(UCSD_TRAIN_PATH)):
        vid_path = os.path.join(UCSD_TRAIN_PATH, vid)
        if not os.path.isdir(vid_path):
            continue

        frames = sorted(os.listdir(vid_path))
        # generate overlapping sequences
        for i in range(len(frames) - SEQ_LEN + 1):
            seq = []
            for j in range(i, i + SEQ_LEN):
                img_path = os.path.join(vid_path, frames[j])
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, INPUT_SHAPE[:2])
                img = img.astype("float32") / 255.0
                seq.append(img)
            if len(seq) == SEQ_LEN:
                X.append(np.array(seq))
                y.append(0)  # training = all normal

    return np.array(X), np.array(y)


def split_dataset(X, y, train_size=0.6):
    """
    Split sequences into train/test sets.
    """
    return train_test_split(X, y, train_size=train_size, random_state=42, shuffle=True)

