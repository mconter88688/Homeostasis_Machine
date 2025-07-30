import constants as cons
import numpy as np
import cv2 
import tensorflow as tf # for TensorFlow
import tensorflow_hub as hub # loads pre-trained feature extraction model from the Hub
from tensorflow.keras.models import Sequential, Model # for model architecture and loading
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, BatchNormalization, Bidirectional, Input, ConvLSTM2D, Conv3DTranspose # for neural network layers
from tensorflow.keras.optimizers import Adam

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
    input_layer = Input(shape = (cons.INPUT_SHAPE))
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

############# HELPER FUNCTIONS #########################################

# Build or load temporal model
def build_model():
    return build_one_way_7_18()


def feature_extractor_setup():
    FEATURE_URL = cons.EFFICIENT_NET_B0
    return hub.KerasLayer(FEATURE_URL, input_shape= cons.INPUT_SHAPE, trainable=False)
    print("Feature extractor loaded!")
# Helper: extract feature from single frame
def extract_feature(frame, feature_extractor):
    resized = cv2.resize(frame, (cons.INPUT_SHAPE[1], cons.INPUT_SHAPE[0])) / 255.0 # resize image and normalize pixel values (originally between 0 and 255) to between 0 and 1
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
    def __init__(self, epochs = 0, batch_size = 0, validation_split = 0, feedback_file = None, model_file = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.feedback_file = feedback_file
        self.model_file = model_file
        self.temp_graph = None

    def redefine_all(self, epochs = 0, batch_size = 0, validation_split = 0, feedback_file = None, model_file = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.feedback_file = feedback_file
        self.model_file = model_file

    