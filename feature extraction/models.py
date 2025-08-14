# models.py
import os
import numpy as np
from collections import deque

import constants as cons

from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, LayerNormalization, BatchNormalization,
    Bidirectional, Input, Conv1D, Conv1DTranspose, RepeatVector, TimeDistributed
)
from tensorflow.keras.optimizers import Adam

# Unified extractor (EfficientNet-B0 for color + small CNNs for HDR/IR)
from feature_extraction import FeatureExtractor


############## MODELS ###########################

def build_simple_7_17():
    """Legacy simple binary classifier over sequences."""
    m = Sequential([
        LSTM(128, input_shape=(cons.SEQ_LEN, cons.FEATURE_DIM), return_sequences=True),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


def build_bi_7_18():
    """Legacy bidirectional LSTM binary classifier."""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(cons.SEQ_LEN, cons.FEATURE_DIM)),
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


def build_one_way_7_18():
    """Legacy one-way LSTM binary classifier."""
    model = Sequential([
        LSTM(128, input_shape=(cons.SEQ_LEN, cons.FEATURE_DIM), return_sequences=True),
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


def build_autoencoder_8_4(seq_len: int = cons.SEQ_LEN,
                          feature_dim: int = cons.FEATURE_DIM,
                          latent_dim: int = 256) -> Model:
    """
    Sequence autoencoder over feature vectors.
    Input:  (B, T=seq_len, F=feature_dim)
    Output: (B, T, F) reconstruction
    """
    inp = Input(shape=(seq_len, feature_dim))

    # --- Encoder ---
    x = Conv1D(512, kernel_size=3, strides=1, padding='same', activation='relu')(inp)
    x = LayerNormalization()(x)

    x = Conv1D(256, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # (B, T/2, 256)
    x = LayerNormalization()(x)

    # Bottleneck
    x = LSTM(latent_dim, return_sequences=False)(x)  # (B, latent_dim)
    x = RepeatVector(seq_len)(x)                     # (B, T, latent_dim)

    # --- Decoder ---
    x = LSTM(256, return_sequences=True)(x)

    x = Conv1DTranspose(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)

    x = Conv1DTranspose(512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)

    out = TimeDistributed(Dense(feature_dim, activation='linear'))(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-4), loss='mse')
    return model


def build_model() -> Model:
    """Current default model: the sequence autoencoder."""
    return build_autoencoder_8_4()


############# TRAINING PARAMS #########################################

class ModelConfigParam:
    def __init__(self, epochs: int = 1, batch_size: int | None = None,
                 validation_split: float = 0.0, feedback_file: str | None = None,
                 model_file: str | None = None, verbose: int = 1,
                 callbacks=None, shuffle: bool = True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.feedback_file = feedback_file
        self.model_file = model_file
        self.temp_graph = None
        self.verbose = verbose
        self.callbacks = callbacks
        self.shuffle = shuffle

    def redefine_all(self, epochs: int = 1, batch_size: int | None = None,
                     validation_split: float = 0.0, feedback_file: str | None = None,
                     model_file: str | None = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.feedback_file = feedback_file
        self.model_file = model_file


############## MAIN WRAPPER ###########################################

class ImageAutoencoder:
    """
    Orchestrates:
      - Unified feature extraction per frameset -> (FEATURE_DIM,) vector
      - Rolling buffer of length SEQ_LEN
      - Autoencoder inference/training
    """

    def __init__(self):
        # Load existing model or build fresh
        if os.path.exists(cons.MODEL_PATH):
            self.model = load_model(cons.MODEL_PATH)
        else:
            self.model = build_model()

        # Unified feature extractor (initialized lazily)
        self.fx: FeatureExtractor | None = None

        # Sequence buffer
        self.buffer = deque(maxlen=cons.SEQ_LEN)

    # -------- Feature Extraction --------

    def feature_extractor_setup(self):
        """
        Backward-compatible initializer. Builds the unified extractor
        (EfficientNet-B0 for color + compact CNNs for HDR/IR).
        """
        if self.fx is None:
            self.fx = FeatureExtractor(trainable_backbones=False)

    def feature_extract_combine(self, frameset) -> np.ndarray:
        """
        frameset: [color_bgr, depth_mm, ir_left, ir_right]
        returns:  (cons.FEATURE_DIM,) vector (expected 1664 if using RGB+HDR+IRL+IRR)
        """
        if self.fx is None:
            self.feature_extractor_setup()
        return self.fx.extract_from_frameset(frameset)

    def feature_append(self, feat: np.ndarray) -> None:
        """Append one (FEATURE_DIM,) vector to the rolling buffer."""
        self.buffer.append(feat)

    def is_buffer_long_enough(self) -> bool:
        return len(self.buffer) == cons.SEQ_LEN

    # -------- Inference / Training --------

    def model_prediction(self) -> float | None:
        """
        Returns a single scalar anomaly score (mean MSE over the window),
        or None if buffer not full.
        """
        if not self.is_buffer_long_enough():
            return None

        seq = np.expand_dims(np.stack(self.buffer), axis=0)  # (1, T, F)
        recon = self.model.predict(seq, verbose=0)           # (1, T, F)
        errors = np.mean((recon[0] - seq[0])**2, axis=1)     # (T,)
        return float(np.mean(errors))

    def fit(self, model_params: ModelConfigParam, train_data_x, train_data_y):
        """
        Train the autoencoder to reconstruct sequences.
        Typically train_data_y == train_data_x for AE training.
        """
        return self.model.fit(
            train_data_x, train_data_y,
            validation_split=model_params.validation_split,
            shuffle=model_params.shuffle,
            epochs=model_params.epochs,
            batch_size=model_params.batch_size,
            callbacks=model_params.callbacks
        )
