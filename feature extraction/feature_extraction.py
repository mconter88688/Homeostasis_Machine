
from __future__ import annotations

import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models

# ---- project constants ----
try:
    import constants as cons
    COLOR_SHAPE = cons.COLOR_INPUT_SHAPE      # (224,224,3)
    IR_SHAPE    = cons.IR_INPUT_SHAPE         # (96,96,1)
    HDR_SHAPE   = cons.HDR_INPUT_SHAPE        # (96,96,1)
    EFF_URL     = cons.EFFICIENT_NET_B0
except Exception:
    # Fallbacks if constants.py isn't available (tests, etc.)
    COLOR_SHAPE = (224, 224, 3)
    IR_SHAPE    = (96, 96, 1)
    HDR_SHAPE   = (96, 96, 1)
    EFF_URL     = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"

COLOR_H, COLOR_W, _ = COLOR_SHAPE
IR_H, IR_W, _       = IR_SHAPE
HDR_H, HDR_W, _     = HDR_SHAPE

COLOR_DIM = 1280
IR_DIM    = 128
HDR_DIM   = 128
COMBINED_DIM = COLOR_DIM + HDR_DIM + 2 * IR_DIM  # 1664

class FeatureExtractor:
    """
    Feature extraction for color/HDR/IR frames.

    - Color: EfficientNet-B0 feature vector (1280-d, frozen by default)
    - HDR:   small Conv2D tower pooled to 128-d
    - IR:    small Conv2D tower pooled to 128-d (shared architecture, separate weights)
    """

    def __init__(self, trainable_backbones: bool = False):
        # Color backbone (EfficientNet-B0 from TF Hub)
        self.color_extractor = hub.KerasLayer(
            EFF_URL,
            input_shape=COLOR_SHAPE,
            trainable=trainable_backbones,
            name="efficientnet_b0_feature_vector",
        )

        # Small CNNs for (H,W,1) inputs (IR / HDR)
        self.ir_extractor  = self._build_small_cnn(IR_SHAPE,  IR_DIM,  name="ir_feature_extractor")
        self.hdr_extractor = self._build_small_cnn(HDR_SHAPE, HDR_DIM, name="hdr_feature_extractor")

        # Sanity check EfficientNet output size (should be 1280)
        dummy = np.zeros(COLOR_SHAPE, np.float32)[None, ...]
        out = self.color_extractor(dummy)
        if int(out.shape[-1]) != COLOR_DIM:
            raise RuntimeError(f"EfficientNet-B0 expected {COLOR_DIM} features, got {int(out.shape[-1])}")

    @staticmethod
    def _build_small_cnn(input_shape, out_dim: int, name: str):
        """Compact Conv net for (H,W,1) inputs -> out_dim vector."""
        inp = layers.Input(shape=input_shape)
        x = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(out_dim, activation='relu')(x)
        return models.Model(inp, x, name=name)

    # ---------- preprocessing helpers ----------

    @staticmethod
    def _prep_color_bgr_to_float_rgb(color_bgr: np.ndarray) -> np.ndarray:
        """
        Resize to (224,224), convert BGR->RGB, scale to [0,1], float32.
        """
        if color_bgr is None or color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError("color_bgr must be an (H,W,3) array")
        img = cv2.resize(color_bgr, (COLOR_W, COLOR_H), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    @staticmethod
    def _prep_single_channel(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """
        Resize (H,W) -> (out_h,out_w,1) and normalize to [0,1].
        If values look like uint8 or millimeters, scale by max for robustness.
        """
        if img is None or img.ndim != 2:
            raise ValueError("single-channel input must be a 2D array (H,W)")
        resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32)
        maxv = float(arr.max()) if arr.size else 0.0
        if maxv > 1.5:
            arr = arr / (maxv + 1e-6)
        return arr[..., None]  # (H,W,1)

    # ---------- extraction APIs ----------

    def extract_color(self, color_bgr: np.ndarray) -> np.ndarray:
        img = self._prep_color_bgr_to_float_rgb(color_bgr)
        vec = self.color_extractor(np.expand_dims(img, 0))
        return tf.squeeze(vec).numpy()  # (1280,)

    def extract_hdr(self, depth_mm: np.ndarray) -> np.ndarray:
        # depth_mm: (H,W), float32 in millimeters (from camera.py)
        hdr = self._prep_single_channel(depth_mm, HDR_H, HDR_W)
        vec = self.hdr_extractor(np.expand_dims(hdr, 0))
        return tf.squeeze(vec).numpy()  # (128,)

    def extract_ir(self, ir_image: np.ndarray) -> np.ndarray:
        # ir_image: (H,W) uint8
        ir = self._prep_single_channel(ir_image, IR_H, IR_W)
        vec = self.ir_extractor(np.expand_dims(ir, 0))
        return tf.squeeze(vec).numpy()  # (128,)

    def extract_from_frameset(self, frameset) -> np.ndarray:
        """
        frameset: [color_bgr, depth_mm, ir_left, ir_right]
        Returns: (COMBINED_DIM,) == 1664 features
        """
        if not isinstance(frameset, (list, tuple)) or len(frameset) != 4:
            raise ValueError("frameset must be [color_bgr, depth_mm, ir_left, ir_right]")

        color_bgr, depth_mm, ir_left, ir_right = frameset

        color_vec = self.extract_color(color_bgr)   # (1280,)
        hdr_vec   = self.extract_hdr(depth_mm)      # (128,)
        irL_vec   = self.extract_ir(ir_left)        # (128,)
        irR_vec   = self.extract_ir(ir_right)       # (128,)

        combined = np.concatenate([color_vec, hdr_vec, irL_vec, irR_vec], axis=0)
        if combined.shape[0] != COMBINED_DIM:
            raise RuntimeError(f"Expected {COMBINED_DIM} features, got {combined.shape[0]}")
        return combined

    # ---------- optional batch API ----------

    def extract_batch(self, framesets: list) -> np.ndarray:
        """
        Process a list of framesets and return (N, COMBINED_DIM) array.
        Each frameset: [color_bgr, depth_mm, ir_left, ir_right]
        """
        feats = [self.extract_from_frameset(fs) for fs in framesets]
        return np.stack(feats, axis=0)

# ---------- quick self-test ----------
if __name__ == "__main__":
    # Fake inputs for a smoke test (no camera required)
    color = np.zeros((480, 640, 3), np.uint8)
    depth = np.full((480, 640), 1000.0, np.float32)  # 1000 mm everywhere
    irL   = np.zeros((480, 640), np.uint8)
    irR   = np.zeros((480, 640), np.uint8)

    fx = FeatureExtractor(trainable_backbones=False)
    vec = fx.extract_from_frameset([color, depth, irL, irR])
    print("Feature vector shape:", vec.shape)  # expect (1664,)
