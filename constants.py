import numpy as np

FEATURE_DIM = 1280
SEQ_LEN = 20                # number of frames in sequence window
IMAGE_MODEL_PATH = "homeostasis_image_model.h5"
LDRD_MODEL_PATH = "homeostatis_ldrd_model.h5"
BEST_MODEL_PATH = "best_model.h5"
FEEDBACK_FILE = "feedback.pkl"
ANOMALY_THRESHOLD = 0.02    # threshold for non-homeostasis
# MOBILE_NET_V2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
EFFICIENT_NET_B0 = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
COLOR_INPUT_SHAPE = (224, 224, 3)
IR_INPUT_SHAPE = (96, 96, 1)
HDR_INPUT_SHAPE = (96, 96, 1)
MODEL_FOLDER = "Models"
DATA_FOLDER = "Data"
WINDOW_NAME = "Camera Output"
MODEL_TYPE = "autoencoder"
STREAM_PATH = "latest.jpg"
RD03D_PORT = "/dev/rd03"
LIDAR_PORT = "/dev/lidar"
TIMEOUT = 1
BLANK_SCREEN = np.zeros((200, 400, 3), dtype=np.uint8)
LIDAR_MAX_POINTS_NUM = 505
RADAR_MAX_TARGETS = 3