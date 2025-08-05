import os
import constants as cons
import numpy as np
import cv2 
import tensorflow as tf # for TensorFlow
import tensorflow_hub as hub # loads pre-trained feature extraction model from the Hub
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model, load_model # for model architecture and loading
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, BatchNormalization, Bidirectional, Input, ConvLSTM2D, Conv3DTranspose, Conv1D, Conv1DTranspose, RepeatVector, TimeDistributed # for neural network layers
from tensorflow.keras.optimizers import Adam
from collections import deque # for sliding window

############## MODELS ###########################

def build_simple_7_17():
    m = Sequential([
        LSTM(128, input_shape=(cons.SEQ_LEN, cons.FEATURE_DIM), return_sequences=True), # return_sequences=True means the full sequence is sent to the next LSTM instead of just the final step
        LSTM(64),
        Dense(32, activation='relu'), # lightweight classifier layer
        Dense(1, activation='sigmoid') # binary classification layer
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_bi_7_18():
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

def build_autoencoder_7_23_not_tested():
    input_layer = Input(shape = (1, cons.SEQ_LEN, 1536))
    x = ConvLSTM2D(32, (3,3), activation = 'relu', padding = 'same', return_sequences = True, strides = (2,2))(input_layer)
    x = LayerNormalization()(x)
    
    x = ConvLSTM2D(64, (3,3), activation='relu', padding = 'same', return_sequences=True, strides=(2,2))(x)
    x = LayerNormalization()(x)

    encoded = ConvLSTM2D(64, (3,3), activation='relu', padding='same', return_sequences=True)(x)

    x = Conv3DTranspose(64, (3,3,3), strides=(1,2,2), padding='same', activation='relu')(encoded)
    x = LayerNormalization()(x)

    x = Conv3DTranspose(32, (3,3,3), strides=(1,2,2), padding='same', activation='relu')(x)
    x = LayerNormalization()(x)

    # decoded should have number of output channels as number of neurons
    decoded = Conv3DTranspose(3, (3,3,3), padding='same', activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer=Adam(1e-4), loss='mse')
    return model

def build_autoencoder_8_4(seq_len=cons.SEQ_LEN, feature_dim=1664, latent_dim=256):
    input_layer = Input(shape=(seq_len, feature_dim))  # (batch, time, features)

    # --- Encoder ---
    x = Conv1D(512, kernel_size=3, strides=1, padding='same', activation='relu')(input_layer)
    x = LayerNormalization()(x)

    x = Conv1D(256, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # (batch, time/2, 256)
    x = LayerNormalization()(x)

    # LSTM Bottleneck
    x = LSTM(latent_dim, return_sequences=False)(x)  # (batch, latent_dim)
    x = RepeatVector(seq_len)(x)  # (batch, time, latent_dim)

    # --- Decoder ---
    x = LSTM(256, return_sequences=True)(x)

    x = Conv1DTranspose(256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)

    x = Conv1DTranspose(512, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)

    decoded = TimeDistributed(Dense(feature_dim, activation='linear'))(x)  # Match original input

    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer=Adam(1e-4), loss='mse')

    return model

############# HELPER FUNCTIONS #########################################

# Build or load temporal model
def build_model():
    return build_autoencoder_8_4()

########### FEATURE EXTRACTORS ###########################################

def build_hdr_feature_extractor(input_shape=cons.HDR_INPUT_SHAPE, output_dim = 128):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(output_dim, activation='relu')(x)  # final feature vector

    return models.Model(inputs, x, name="depth_feature_extractor")

def build_ir_feature_extractor(input_shape= cons.IR_INPUT_SHAPE, output_dim=128):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),  # makes output shape (batch_size, 64)

        layers.Dense(output_dim, activation='relu')  # bottleneck
    ])
    return model

def build_color_feature_extractor():
    FEATURE_URL = cons.EFFICIENT_NET_B0
    print("Feature extractor loaded!")
    return hub.KerasLayer(FEATURE_URL, input_shape= cons.COLOR_INPUT_SHAPE, trainable=False)
    
# Helper: extract feature from single frame
def color_extract_feature(frame, feature_extractor):
    resized = cv2.resize(frame, (cons.COLOR_INPUT_SHAPE[1], cons.COLOR_INPUT_SHAPE[0])) / 255.0 # resize image and normalize pixel values (originally between 0 and 255) to between 0 and 1
    tensor = tf.expand_dims(resized.astype(np.float32), axis=0) # add batch dimension and convert numbers to floats
    feats = feature_extractor(tensor) # use feature extractor on adjusted frame
    return tf.squeeze(feats).numpy()  # shape (1280,), NumPy array

def model_prediction(model, sequence, type):
    if type == "autoencoder":
        reconstruction = model.predict(sequence)
        errors = np.mean((reconstruction - sequence)**2, axis=(1,2,3,4)) # average errors across all dimensions to get idea of how model did overall
        return errors
    else:
        return model.predict(sequence, verbose=0)[0][0]

############## CLASSES ###############################################

class ModelConfigParam:
    def __init__(self, epochs = 1, batch_size = None, validation_split = 0.0, feedback_file = None, model_file = None, verbose = 1, callbacks=None, shuffle = True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.feedback_file = feedback_file
        self.model_file = model_file
        self.temp_graph = None 
        self.verbose=verbose
        self.callbacks=callbacks,
        self.shuffle=shuffle

    def redefine_all(self, epochs = 1, batch_size = None, validation_split = 0.0, feedback_file = None, model_file = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.feedback_file = feedback_file
        self.model_file = model_file


class Autoencoder:
    def __init__(self):
        if os.path.exists(cons.MODEL_PATH):
            self.model = load_model(cons.MODEL_PATH) # function imported from tensorflow.keras.models
        else:
            self.model = build_model()
        self.color_feature_extractor = None
        self.ir_feature_extractor = None
        self.hdr_feature_extractor = None
        self.buffer = deque(maxlen=cons.SEQ_LEN)

        

    def feature_extractor_setup(self): 
        self.color_feature_extractor = build_color_feature_extractor()
        self.ir_feature_extractor = build_ir_feature_extractor()
        self.hdr_feature_extractor = build_hdr_feature_extractor()

    def extract_color_features(self, frameset):
        #frameset is color_image, merged_depth_in_mm, ir_left, ir_right
        
        
        color_resized = cv2.resize(frameset[0], (cons.COLOR_INPUT_SHAPE[1], cons.COLOR_INPUT_SHAPE[0])) / 255.0 # resize image and normalize pixel values (originally between 0 and 255) to between 0 and 1
        color_tensor = tf.expand_dims(color_resized.astype(np.float32), axis=0) # add batch dimension and convert numbers to floats
        color_feats = self.color_feature_extractor(color_tensor) # use feature extractor on adjusted frame
        return tf.squeeze(color_feats).numpy()  # shape (1280,), NumPy array
        
    def extract_ir_features(self, frameset, left_or_right):
        ir_resized = cv2.resize(frameset[left_or_right+2], (cons.IR_INPUT_SHAPE[1], cons.IR_INPUT_SHAPE[0]), interpolation=cv2.INTER_AREA) / 255.0
        ir_resized = ir_resized[..., np.newaxis]

        ir_tensor = tf.expand_dims(ir_resized.astype(np.float32), axis=0)
        ir_feats = self.ir_feature_extractor(ir_tensor)
        final_feats =  tf.squeeze(ir_feats).numpy()
        #print(final_feats)
        return final_feats

    
    def extract_hdr_features(self, frameset):
        hdr_resized = cv2.resize(frameset[1], (cons.HDR_INPUT_SHAPE[1], cons.HDR_INPUT_SHAPE[0]), interpolation=cv2.INTER_AREA) / 255.0
        hdr_resized = hdr_resized[..., np.newaxis]

        hdr_tensor = tf.expand_dims(hdr_resized.astype(np.float32), axis=0)

        hdr_feats = self.hdr_feature_extractor(hdr_tensor)
        final_feats = tf.squeeze(hdr_feats).numpy()
        #print(final_feats.shape)
        return final_feats
    
    def feature_extract_combine(self, frameset):
        color_features = self.extract_color_features(frameset)
        hdr_features = self.extract_hdr_features(frameset)
        left_ir_features = self.extract_ir_features(frameset, 0)
        right_ir_features = self.extract_ir_features(frameset, 1)

        combined = np.concatenate([color_features, hdr_features,left_ir_features, right_ir_features], axis=0)
        #print(combined.shape)
        return combined
        
    def feature_append(self, feat):
        self.buffer.append(feat)

    def model_prediction(self):
        if not len(self.buffer) == cons.SEQ_LEN:
            return None
        else:
            seq = np.expand_dims(np.stack(self.buffer), axis=0)  # shape (1,SEQ_LEN,FEATURE_DIM)
            # Get reconstruction from model
            reconstruction = self.model.predict(seq)  # shape: (1, 20, 1536)

            # Compute per-timestep MSE
            errors = np.mean((reconstruction[0] - seq[0])**2, axis=1)  # shape: (20,)

            return np.mean(errors)
        
    def fit(self, model_params, train_data_x, train_data_y):
        self.model.fit(train_data_x, train_data_y, validation_split = model_params.validation_split,shuffle=model_params.shuffle, epochs=model_params.epochs, batch_size=model_params.batch_size, callbacks=model_params.callbacks)

    def is_buffer_long_enough(self):
        return len(self.buffer) == cons.SEQ_LEN
        
    

    