from . import constants as cons
from models import build_bi_7_18
from . import dataset_utils_kaggle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

print("Loading UCSD Ped2 Kaggle dataset...")
X, y = load_ucsd_sequences()
print(f"Loaded {len(y)} sequences")

X_train, X_test, y_train, y_test = split_dataset(X, y, train_size=0.6)
print(f"Training on {len(y_train)} sequences, testing on {len(y_test)} sequences")

# Build model
model = build_bi_7_18()

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(cons.BEST_MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# Save model
model.save(cons.MODEL_PATH)
print(f"Model saved at {cons.MODEL_PATH}")

# Evaluate
y_prob = model.predict(X_test, verbose=0).squeeze()
y_pred = (y_prob > cons.ANOMALY_THRESHOLD).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Test Accuracy: {acc:.3f}, AUC: {auc:.3f}")
