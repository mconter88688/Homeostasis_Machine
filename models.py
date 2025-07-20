import constants as cons
from tensorflow.keras.models import Sequential # for model architecture and loading
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional # for neural network layers



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


class ModelConfigParam:
    def __init__(self, epochs = 0, batch_size = 0, validation_split = 0, feedback_file = None, model_file = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.feedback_file = feedback_file
        self.model_file = model_file

    