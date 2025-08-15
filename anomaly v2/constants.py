import numpy as np

# === Feature / Sequence Settings ===
FEATURE_DIM = 1280            # EfficientNet-B0 RGB features only
SEQ_LEN = 20                  # Frames per sequence window

# === Model / Feedback Paths ===
MODEL_PATH = "homeostasis_model.h5"
BEST_MODEL_PATH = "best_model.h5"
FEEDBACK_FILE = "feedback.pkl"

# === Anomaly Detection ===
ANOMALY_THRESHOLD = 0.02      # Default, can be overridden by learned threshold

# === Pretrained Feature Extractor URL ===
# MOBILE_NET_V2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
EFFICIENT_NET_B0 = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"

# === Input Shapes ===
COLOR_INPUT_SHAPE = (224, 224, 3)  # RGB frames
IR_INPUT_SHAPE = (96, 96, 1)       # For live Jetson sensors (not used in Ped2)
HDR_INPUT_SHAPE = (96, 96, 1)      # For live Jetson sensors (not used in Ped2)

# === Folders ===
MODEL_FOLDER = "Models"
DATA_FOLDER = "Data"

# === Display & Misc ===
WINDOW_NAME = "Camera Output"
MODEL_TYPE = "autoencoder"

# === Jetson Hardware Ports (kept for compatibility) ===
STREAM_PATH = "latest.jpg"
RD03D_PORT = "/dev/rd03"
LIDAR_PORT = "/dev/lidar"
TIMEOUT = 1
BLANK_SCREEN = np.zeros((200, 400, 3), dtype=np.uint8)
