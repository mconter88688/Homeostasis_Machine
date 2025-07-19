import constants as cons
from tensorflow.keras.models import Sequential, load_model # for model architecture and loading
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