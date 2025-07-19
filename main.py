import fsm as fsm
import cv2 # for camera
import numpy as np # for arrays
import tensorflow as tf # for TensorFlow
import tensorflow_hub as hub # loads pre-trained feature extraction model from the Hub
from tensorflow.keras.models import Sequential, load_model # for model architecture and loading
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional # for neural network layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import deque # for sliding window
import os # file and directory management
import pickle
import constants as cons
import models as mod

# CONFIGURATION
print("Starting configuration!")
#CAM_INDEX              # USB camera index

print("Configuration done!")

buffer = deque(maxlen=cons.SEQ_LEN)



### SETUP ###
# Load previous feedback data if it exists
if os.path.exists(cons.FEEDBACK_FILE):
    try:
        with open(cons.FEEDBACK_FILE, "rb") as f:
            NORMAL_DATA, ANOMALY_DATA = pickle.load(f)
    except EOFError:
        NORMAL_DATA, ANOMALY_DATA = [], []
else:
    NORMAL_DATA, ANOMALY_DATA = [], []
print("Feedback file loaded")

# Load visual feature extractor
FEATURE_URL = cons.EFFICIENT_NET_B0
feature_extractor = hub.KerasLayer(FEATURE_URL, input_shape= cons.INPUT_SHAPE, trainable=False)
print("Feature extractor loaded")

# Build or load temporal model
def build_model():
    # m = Sequential([
    #     LSTM(128, input_shape=(SEQ_LEN, FEATURE_DIM), return_sequences=True), # return_sequences=True means the full sequence is sent to the next LSTM instead of just the final step
    #     LSTM(64),
    #     Dense(32, activation='relu'), # lightweight classifier layer
    #     Dense(1, activation='sigmoid') # binary classification layer
    # ])
    # m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # return m
    # model = Sequential([
    #     Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQ_LEN, FEATURE_DIM)),
    #     BatchNormalization(),
    #     Dropout(0.3),
    #     LSTM(64),
    #     BatchNormalization(),
    #     Dropout(0.3),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # return model
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

if os.path.exists(cons.MODEL_PATH):
    temporal_model = load_model(cons.MODEL_PATH) # function imported from tensorflow.keras.models
else:
    temporal_model = build_model()

# Helper: extract feature from single frame
def extract_feature(frame):
    resized = cv2.resize(frame, (cons.INPUT_SHAPE[1], cons.INPUT_SHAPE[0])) / 255.0 # resize image and normalize pixel values (originally between 0 and 255) to between 0 and 1
    tensor = tf.expand_dims(resized.astype(np.float32), axis=0) # add batch dimension and convert numbers to floats
    feats = feature_extractor(tensor) # use feature extractor on adjusted frame
    return tf.squeeze(feats).numpy()  # shape (1280,), NumPy array

for cam in range(5):
    cap = cv2.VideoCapture(cam)
    if cap.isOpened():
        CAM_INDEX = cam
        break
else:
    raise RuntimeError("No USB camera found.")  
print("Camera Found!")


class NormalDataTraining(fsm.State):
    def __init__(self, FSM, NORMAL_DATA):
        self.FSM = FSM
        self.num_frames = 0
        self.NORMAL_DATA = NORMAL_DATA

    def Enter(self):
        print("Normal Feedback Data Mode")
        self.num_frames = 0

    
    def Execute(self):
        # Train on only normal feedback
        ret, frame = cap.read()
        if not ret:
            print("Unsuccessful frame capture. Going to Menu...")
            self.FSM.Transition("toMenu")
            return
        feat = extract_feature(frame)
        buffer.append(feat)
        if len(buffer) == cons.SEQ_LEN:
            self.NORMAL_DATA.append(np.stack(buffer))
        
        cv2.putText(frame, f"{self.num_frames}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow("Anomaly Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.FSM.Transition("toMenu")
        self.num_frames+=1


    def Exit(self):
        buffer.clear()
        cv2.destroyAllWindows()

class WipingModelAndFeedback(fsm.State):
    def __init__(self, FEEDBACK_FILE, MODEL_PATH, FSM, NORMAL_DATA, ANOMALY_DATA):
        self.FEEDBACK_FILE = FEEDBACK_FILE
        self.MODEL_PATH = MODEL_PATH
        self.FSM = FSM
        self.NORMAL_DATA = NORMAL_DATA
        self.ANOMALY_DATA = ANOMALY_DATA

    def Enter(self):
        print("Deleting model and feedback data")
    
    def Execute(self):
        self.NORMAL_DATA.clear()
        self.ANOMALY_DATA.clear()
        
        with open(cons.FEEDBACK_FILE, "wb") as f:
            pass

        if os.path.exists(self.MODEL_PATH):
            os.remove(self.MODEL_PATH)
        print("Feedback and model successfully removed!")
        self.FSM.Transition("toMenu")
        return
    def Exit(self):
        pass
        
    

    
class Menu(fsm.State):
    def __init__(self, FSM):
        self.FSM = FSM
    
    def Enter(self):
        print("**Select from the following:**")
        print("Wipe Model and Feedback:...........................W")
        print("Save Feedback and Retrain Model:...................S")
        print("Take in Normal Training Data:......................N")
        print("Give User Input on Normal and Abnormal Scenes:.....F")

            
    def Execute(self):
        answer = input("").strip().upper()
        if answer == "W":
            self.FSM.Transition("toWipingModelAndFeedback")
        elif answer == "S":
            self.FSM.Transition("toSavingModelAndFeedback")
        elif answer == "N":
            self.FSM.Transition("toNormalDataTraining")
        elif answer == "F":
            self.FSM.Transition("toRLHF")
        else:
            print("Invalid input. Try again.")

    def Exit(self):
        pass


class SavingModelAndFeedback(fsm.State):
    def __init__(self, FEEDBACK_FILE, MODEL_PATH, FSM, NORMAL_DATA, ANOMALY_DATA):
        self.FEEDBACK_FILE = FEEDBACK_FILE
        self.MODEL_PATH = MODEL_PATH
        self.FSM = FSM
        self.NORMAL_DATA = NORMAL_DATA
        self.ANOMALY_DATA = ANOMALY_DATA
    
    def Enter(self):
        print("Saving Model and Feedback File")

    def Execute(self):
        if self.NORMAL_DATA or self.ANOMALY_DATA:
            print("Retraining model with feedback data...")
            # Create training sets
            X = np.array(self.NORMAL_DATA + self.ANOMALY_DATA)
            y = np.array([0]*len(self.NORMAL_DATA) + [1]*len(self.ANOMALY_DATA)) # trains it with predictions being certain of normal v.s. anomaly scenarios
            temporal_model.fit(X, y, epochs=5, batch_size=4)
            temporal_model.save(cons.MODEL_PATH)
            print("Model updated and saved.")

        with open(cons.FEEDBACK_FILE, "wb") as f:
            pickle.dump((self.NORMAL_DATA, self.ANOMALY_DATA), f)
        
        self.FSM.Transition("toMenu")
        return

    def Exit(self):
        pass

class RLHF(fsm.State):
    def __init__(self, FSM, NORMAL_DATA, ANOMALY_DATA):
        self.FSM = FSM
        self.NORMAL_DATA = NORMAL_DATA
        self.ANOMALY_DATA = ANOMALY_DATA

    def Enter(self):
        print("Human Feedback Mode")
        print("Press 'n' to label homeostasis, 'a' to label abnormalities, and 'q' to quit.")

    def Execute(self):
        ret, frame = cap.read()
        if not ret:
            print("Unsuccessful frame capture. Going to Menu...")
            self.FSM.Transition("toMenu")
            return
        feat = extract_feature(frame)
        buffer.append(feat) # add 1D array to end of the buffer

        label = "N/A"
        if len(buffer) == cons.SEQ_LEN:
            seq = np.expand_dims(np.stack(buffer), axis=0)  # shape (1,SEQ_LEN,FEATURE_DIM)
            pred = temporal_model.predict(seq, verbose=0)[0][0] # gets the number spit out by the temporal model
            is_anomaly = pred > cons.ANOMALY_THRESHOLD #checks prediction against threshold
            label = f"{'ANOMALY' if is_anomaly else 'NORMAL'} ({pred:.2f})"
            color = (0,0,255) if is_anomaly else (0,255,0)

            # Draw on frame
            cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Anomaly Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.FSM.Transition("toMenu")
            return
        elif key == ord('n') and len(buffer) == cons.SEQ_LEN:
            self.NORMAL_DATA.append(np.stack(buffer))
            print("Labeled one normal sequence")
        elif key == ord('a') and len(buffer) == cons.SEQ_LEN:
            self.ANOMALY_DATA.append(np.stack(buffer))
            print("Labeled one anomalous sequence")

    def Exit(self):
        buffer.clear()
        cv2.destroyAllWindows()


print("About to make HS_MODEL")
hs_model = fsm.HS_Model()
hs_model.FSM.states["NormalDataTraining"] = NormalDataTraining(hs_model.FSM, NORMAL_DATA)
hs_model.FSM.states["RLHF"] = RLHF(hs_model.FSM, NORMAL_DATA, ANOMALY_DATA)
hs_model.FSM.states["SavingModelAndFeedback"] = SavingModelAndFeedback(cons.FEEDBACK_FILE, cons.MODEL_PATH, hs_model.FSM, NORMAL_DATA, ANOMALY_DATA)
hs_model.FSM.states["WipingModelAndFeedback"] = WipingModelAndFeedback(cons.FEEDBACK_FILE, cons.MODEL_PATH, hs_model.FSM, NORMAL_DATA, ANOMALY_DATA)
hs_model.FSM.states["Menu"] = Menu(hs_model.FSM)
hs_model.FSM.transitions["toMenu"] = fsm.Transition("Menu")
hs_model.FSM.transitions["toNormalDataTraining"] = fsm.Transition("NormalDataTraining")
hs_model.FSM.transitions["toRLHF"] = fsm.Transition("RLHF")
hs_model.FSM.transitions["toSavingModelAndFeedback"] = fsm.Transition("SavingModelAndFeedback")
hs_model.FSM.transitions["toWipingModelAndFeedback"] = fsm.Transition("WipingModelAndFeedback")

hs_model.FSM.Transition("toMenu")
print("About to execute HS_MODEL")
while True:
    hs_model.FSM.Execute()