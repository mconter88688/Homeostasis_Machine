import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import deque
import os
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

FEATURE_DIM = 1280
SEQ_LEN = 20
TOTAL_FRAMES = 1000
MODEL_PATH = "homeostasis_model.h5"
BEST_MODEL_PATH = "best_model.h5"
FEEDBACK_FILE = "feedback.pkl"
ANOMALY_THRESHOLD = 0.6
EFFICIENT_NET_B0 = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
INPUT_SHAPE = (224, 224, 3)

# Load previous feedback data if it exists
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "rb") as f:
        NORMAL_DATA, ANOMALY_DATA = pickle.load(f)
else:
    NORMAL_DATA, ANOMALY_DATA = [], []

# Load visual feature extractor
feature_extractor = hub.KerasLayer(EFFICIENT_NET_B0, input_shape=INPUT_SHAPE, trainable=False)

# Build improved temporal model
def build_model():
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQ_LEN, FEATURE_DIM)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load or initialize model
if os.path.exists(MODEL_PATH):
    temporal_model = load_model(MODEL_PATH)
else:
    temporal_model = build_model()

# Feature extraction helper
def extract_feature(frame):
    resized = cv2.resize(frame, (INPUT_SHAPE[1], INPUT_SHAPE[0])) / 255.0
    tensor = tf.expand_dims(resized.astype(np.float32), axis=0)
    feats = feature_extractor(tensor)
    return tf.squeeze(feats).numpy()

# Get camera index
for cam in range(5):
    cap = cv2.VideoCapture(cam)
    if cap.isOpened():
        CAM_INDEX = cam
        break
else:
    raise RuntimeError("No USB camera found.")

# Collect initial training data (normal sequences)
buffer = deque(maxlen=SEQ_LEN)
num_frames = 0
while num_frames < TOTAL_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    feat = extract_feature(frame)
    buffer.append(feat)
    if len(buffer) == SEQ_LEN:
        NORMAL_DATA.append(np.stack(buffer))
    num_frames += 1

# Train on normal data
X = np.array(NORMAL_DATA)
y = np.array([0] * len(NORMAL_DATA))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
]

history = temporal_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=4,
    callbacks=callbacks,
    verbose=1
)

temporal_model.save(MODEL_PATH)

# Plot training & validation loss and accuracy
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Live prediction and feedback loop
print("Press 'n' to label homeostasis, 'a' for anomaly, 'q' to quit.")
buffer = deque(maxlen=SEQ_LEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feat = extract_feature(frame)
    buffer.append(feat)

    label = "N/A"
    if len(buffer) == SEQ_LEN:
        seq = np.expand_dims(np.stack(buffer), axis=0)
        pred = temporal_model.predict(seq, verbose=0)[0][0]
        is_anomaly = pred > ANOMALY_THRESHOLD
        label = f"{'ANOMALY' if is_anomaly else 'NORMAL'} ({pred:.2f})"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Anomaly Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n') and len(buffer) == SEQ_LEN:
        NORMAL_DATA.append(np.stack(buffer))
        print("Labeled one normal sequence")
    elif key == ord('a') and len(buffer) == SEQ_LEN:
        ANOMALY_DATA.append(np.stack(buffer))
        print("Labeled one anomalous sequence")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Retrain if new data was labeled
if NORMAL_DATA or ANOMALY_DATA:
    print("Retraining with labeled feedback...")
    X = np.array(NORMAL_DATA + ANOMALY_DATA)
    y = np.array([0]*len(NORMAL_DATA) + [1]*len(ANOMALY_DATA))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    history = temporal_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=4,
        callbacks=callbacks,
        verbose=1
    )

    temporal_model.save(MODEL_PATH)

    # Plot updated training
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs (Retrained)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Over Epochs (Retrained)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Model updated and saved.")

# Save feedback
with open(FEEDBACK_FILE, "wb") as f:
    pickle.dump((NORMAL_DATA, ANOMALY_DATA), f)

