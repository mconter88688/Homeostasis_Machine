import fsm as fsm
import cv2 # for camera
import numpy as np # for arrays
import tensorflow as tf # for TensorFlow
import tensorflow_hub as hub # loads pre-trained feature extraction model from the Hub
from tensorflow.keras.models import load_model # for model architecture and loading
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import deque # for sliding window
import os # file and directory management
import pickle
import constants as cons
import models as mod
#from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import camera as cam

# CONFIGURATION
print("Starting configuration!")
#CAM_INDEX              # USB camera index

print("Configuration done!")

buffer = deque(maxlen=cons.SEQ_LEN)

class Data:
    def __init__(self):
        self.normal_data = []
        self.anomaly_data = []
    
    def load_data(self, feedback_file):
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, "rb") as f:
                    self.normal_data, self.anomaly_data = pickle.load(f)
            except EOFError:
                self.normal_data, self.anomaly_data = [], []
                print("File is empty")
        else:
            self.normal_data, self.anomaly_data = [], []
            print("file does not exist")

    def clear_data(self):
        self.anomaly_data.clear()
        self.normal_data.clear()

    def append_normal_data(self, new_data):
        self.normal_data.append(new_data)
    
    def append_anomaly_data(self, new_data):
        self.anomaly_data.append(new_data)

    def save_data(self, feedback_file):
        with open(feedback_file, "wb") as f:
            pickle.dump((self.normal_data, self.anomaly_data), f)

    def is_empty(self):
        return (len(self.normal_data) == 0 and len(self.anomaly_data) == 0)
    

### SETUP ###
# Load previous feedback data if it exists

model_data = Data()
model_data.load_data(cons.FEEDBACK_FILE)
print("Feedback file loaded")

# Class instance for model parameters
model_params = mod.ModelConfigParam(5, 4)

# Ensure the model folder is in the directory
model_folder_path = os.path.join(os.getcwd(), cons.MODEL_FOLDER)
if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)
print("Model folder exists!")

# Ensure the data folder is in the directory
data_folder_path = os.path.join(os.getcwd(), cons.DATA_FOLDER)
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
print("Feedback folder exists!")


# Load visual feature extractor
FEATURE_URL = cons.EFFICIENT_NET_B0
feature_extractor = hub.KerasLayer(FEATURE_URL, input_shape= cons.INPUT_SHAPE, trainable=False)
print("Feature extractor loaded")

# Build or load temporal model
def build_model():
    return mod.build_one_way_7_18()

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

camera = cam.Camera()
camera.configure_streams()
camera.configure_HDR()
camera.start()
# cv2.namedWindow(cons.WINDOW_NAME, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(cons.WINDOW_NAME, 1280, 960)  # Adjusted for 2x2 layout


class NormalDataTraining(fsm.State):
    def __init__(self, FSM, model_data):
        self.FSM = FSM
        self.num_frames = 0
        self.model_data = model_data

    def Enter(self):
        print("Normal Feedback Data Mode")
        camera.number = 0
        camera.mode = "NormalDataTraining"

    
    def Execute(self):
        # Train on only normal feedback
        ret, frame, processed_frames = camera.one_capture()
        if not ret:
            # print("Unsuccessful frame capture. Going to Menu...")
            # self.FSM.Transition("toMenu")
            return
        feat = extract_feature(frame[0])
        buffer.append(feat)
        if len(buffer) == cons.SEQ_LEN:
            self.model_data.append_normal_data(np.stack(buffer))
        # Create and display the combined view
        display = camera.create_display(processed_frames)
        cv2.imshow("Normal Training Views", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.FSM.Transition("toMenu")
        camera.number+=1


    def Exit(self):
        buffer.clear()
        cv2.destroyAllWindows()

class WipingModelAndFeedback(fsm.State):
    def __init__(self, FEEDBACK_FILE, MODEL_PATH, FSM, model_data):
        self.FEEDBACK_FILE = FEEDBACK_FILE
        self.MODEL_PATH = MODEL_PATH
        self.FSM = FSM
        self.model_data = model_data

    def Enter(self):
        print("Deleting model and feedback data")
    
    def Execute(self):
        self.model_data.clear_data()
        
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
        print("Retrain Model:.....................................R")
        print("Take in Normal Training Data:......................N")
        print("Give User Input on Normal and Abnormal Scenes:.....F")
        print("Document Your Currently Loaded Model...............M")
        print("Document Your Currently Loaded Data................D")
        print("Load in a Saved Model..............................L")

            
    def Execute(self):
        answer = input("").strip().upper()
        if answer == "W":
            self.FSM.Transition("toWipingModelAndFeedback")
        elif answer == "R":
            self.FSM.Transition("toSavingModelAndFeedback")
        elif answer == "N":
            self.FSM.Transition("toNormalDataTraining")
        elif answer == "F":
            self.FSM.Transition("toRLHF")
        elif answer == "M":
            self.FSM.Transition("toDocumentModel")
        elif answer == "L":
            self.FSM.Transition("toLoadModel")
        elif answer == "D":
            self.FSM.Transition("toDocumentFeedback")
        else:
            print("Invalid input. Try again.")

    def Exit(self):
        pass


class SavingModelAndFeedback(fsm.State):
    def __init__(self, FEEDBACK_FILE, MODEL_PATH, FSM, model_data, model_params):
        self.FEEDBACK_FILE = FEEDBACK_FILE
        self.MODEL_PATH = MODEL_PATH
        self.FSM = FSM
        self.model_data = model_data
        self.model_params = model_params
    
    def Enter(self):
        print("Saving Model and Feedback File")

    def Execute(self):
        # Retrain model
        callbacks = [
                        EarlyStopping(patience=3, restore_best_weights=True),
                        ModelCheckpoint(cons.BEST_MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
                    ]
        epoch_num = int(input("Epochs: "))
        batch_num = int(input("Batch Size: "))
        validation_num = float(input("Validation Split: "))
        self.model_params.redefine_all(epoch_num, batch_num, validation_num, None, None)
        answer = input("Would you like to load a saved data file?").strip().upper()
        ## TODO: Complete this load data file thing
        if answer == "Y":
            print("Available data folders:")
            for folder in os.listdir(cons.DATA_FOLDER):
                print("-", folder)
            if not os.listdir(cons.DATA_FOLDER):
                print("DATA_FOLDER is empty.")
            else:
                good_data = False
                while not good_data:
                    answer = input("Select data to load: ")
                    if answer in os.listdir(cons.DATA_FOLDER):
                        good_data = True
                        data_path = os.path.join(os.getcwd(), cons.DATA_FOLDER, answer, answer + ".pkl")
                        self.model_data.load_data(data_path)
                    elif answer.upper() == "Q":
                        break
                    else:
                        print("Data file does not exist. Try again")
            
        # Save feedback into file
        self.model_data.save_data(cons.FEEDBACK_FILE)
        
        if not self.model_data.is_empty():
            print("Retraining model with feedback data...")
            # Create training sets
            X = np.array(self.model_data.normal_data + self.model_data.anomaly_data)
            y = np.array([0]*len(self.model_data.normal_data) + [1]*len(self.model_data.anomaly_data)) # trains it with predictions being certain of normal v.s. anomaly scenarios
            history = temporal_model.fit(X, y, validation_split = self.model_params.validation_split, shuffle=True, epochs=self.model_params.epochs, batch_size=self.model_params.batch_size, callbacks=callbacks)
            temporal_model.save(cons.MODEL_PATH)
            print("Model updated and saved.")

            answer = input("Would you like to graph the data? (Y/N)").strip().upper()
            if answer == "Y":
                # Plot training & validation loss and accuracy
                plt.figure(figsize=(10, 4))

                plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.title('Loss Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(history.history['accuracy'], label='Train Acc')
                plt.plot(history.history['val_accuracy'], label='Val Acc')
                plt.title('Accuracy Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()

                plt.tight_layout()

                graph_path = os.path.join(os.getcwd(), "temp_training_plot.png")
                plt.savefig(graph_path)

                img = cv2.imread(graph_path)
                if img is not None:
                    cv2.imshow("Training Plot", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                answer = input("Would you like to save this model? (Y/N)").strip().upper()
                if answer == "Y":
                    
                    self.model_params.temp_graph = graph_path
                    
                    self.FSM.Transition("toDocumentModel")
                    return

        self.FSM.Transition("toMenu")
        return

    def Exit(self):
        pass


class LoadModel(fsm.State):
    def __init__(self, FSM, model_params, temporal_model):
        self.FSM = FSM
        self.model_params = model_params
        self.temporal_model = temporal_model

    def Enter(self):
        print("Available model folders:")
        for folder in os.listdir(cons.MODEL_FOLDER):
            print("-", folder)

    def Execute(self):
        if not os.listdir(cons.MODEL_FOLDER):
            print("MODEL_FOLDER is empty.")
            self.FSM.Transition("toMenu")
            return
        good_model = False
        while not good_model:
            answer = input("Select model to load: ").replace(" ", "")
            if answer.upper() == "Q":
                print("Quitting...")
                self.FSM.Transition("toMenu")
                return
            if answer in os.listdir(cons.MODEL_FOLDER):
                good_model = True
                self.temporal_model = load_model(os.path.join(os.getcwd(), cons.MODEL_FOLDER, answer, answer + ".h5"))
                self.model_params.epochs = 0
                self.model_params.batch_size = 0
                self.model_params.validation_split = 0
                self.model_params.model_file = answer + ".h5"
                self.model_params.feedback_file = None
            else:
                print("Model does not exist. Try again")
        self.FSM.Transition("toMenu")

    def Exit(self):
        pass

## TODO: Finish document feedback
class DocumentFeedback(fsm.State):
    def __init__(self, FSM, model_params, model_data):
        self.FSM = FSM
        self.model_params = model_params
        self.model_data = model_data
    
    def Enter(self):
        pass

    def Execute(self):
        good_file = False
        while not good_file:
            answer = input("Name of data file: ").replace(" ", "")
            if answer.upper() == "Q":
                print("Quitting...")
                self.FSM.Transition("toMenu")
                return
            file_name = answer + ".pkl"
            folder_path = os.path.join(os.getcwd(), cons.DATA_FOLDER, answer)
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(folder_path):
                print("Data file name already exists. Try again.")
            else:
                good_file = True
                os.makedirs(folder_path)
        self.model_data.save_data(cons.FEEDBACK_FILE)
        self.model_data.save_data(file_path)
        info_path = os.path.join(folder_path, "info.txt")
        self.model_params.feedback_file = file_path
        notes = input("Notes: ")
        with open(info_path, 'w') as f:
            f.write(notes)
        self.FSM.Transition("toMenu")
        pass

    def Exit(self):
        pass


class DocumentModel(fsm.State):
    def __init__(self, FSM, model_params, temporal_model):
        self.FSM = FSM
        self.model_params = model_params
        self.temporal_model = temporal_model

    def Enter(self):
        pass

    def Execute(self):
        good_file = False
        while not good_file:
            answer = input("Name of model: ").replace(" ", "")
            if answer.upper() == "Q":
                print("Quitting...")
                self.FSM.Transition("toMenu")
                return
            file_name = answer + ".h5"
            folder_path = os.path.join(os.getcwd(), cons.MODEL_FOLDER, answer)
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(folder_path):
                print("Model name already exists. Try again.")
            else:
                good_file = True
                os.makedirs(folder_path)
        self.temporal_model.save(file_path)
        if (self.model_params.temp_graph != None) and os.path.exists(self.model_params.temp_graph):
            graph_target = os.path.join(folder_path, "training_plot.png")
            os.rename(self.model_params.temp_graph, graph_target)
            self.model_params.temp_graph = None
        info_path = os.path.join(folder_path, "info.txt")
        notes = input("Notes: ")
        with open(info_path, 'w') as f:
            f.write(f"Model Training Info\n===================\n")
            f.write(f"Epochs:            {self.model_params.epochs}\n")
            f.write(f"Batch Size:        {self.model_params.batch_size}\n")
            f.write(f"Validation Split:  {self.model_params.validation_split}\n")
            #f.write(f"Feedback File:     {model_params.feedback_file}\n")
            f.write("\n")
            f.write(notes)
        self.FSM.Transition("toMenu")


    def Exit(self):
        print("Model Saved!")


class RLHF(fsm.State):
    def __init__(self, FSM, model_data):
        self.FSM = FSM
        self.model_data = model_data

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
            self.model_data.append_normal_data(np.stack(buffer))
            print("Labeled one normal sequence")
        elif key == ord('a') and len(buffer) == cons.SEQ_LEN:
            self.model_data.append_anomaly_data(np.stack(buffer))
            print("Labeled one anomalous sequence")

    def Exit(self):
        buffer.clear()
        cv2.destroyAllWindows()


print("About to make HS_MODEL")
hs_model = fsm.HS_Model()
hs_model.FSM.states["NormalDataTraining"] = NormalDataTraining(hs_model.FSM, model_data)
hs_model.FSM.states["RLHF"] = RLHF(hs_model.FSM, model_data)
hs_model.FSM.states["SavingModelAndFeedback"] = SavingModelAndFeedback(cons.FEEDBACK_FILE, cons.MODEL_PATH, hs_model.FSM, model_data, model_params)
hs_model.FSM.states["WipingModelAndFeedback"] = WipingModelAndFeedback(cons.FEEDBACK_FILE, cons.MODEL_PATH, hs_model.FSM, model_data)
hs_model.FSM.states["Menu"] = Menu(hs_model.FSM)
hs_model.FSM.states["DocumentModel"] = DocumentModel(hs_model.FSM, model_params, temporal_model)
hs_model.FSM.states["LoadModel"] = LoadModel(hs_model.FSM, model_params, temporal_model)
hs_model.FSM.states["DocumentFeedback"] = DocumentFeedback(hs_model.FSM, model_params, model_data)
hs_model.FSM.transitions["toMenu"] = fsm.Transition("Menu")
hs_model.FSM.transitions["toNormalDataTraining"] = fsm.Transition("NormalDataTraining")
hs_model.FSM.transitions["toRLHF"] = fsm.Transition("RLHF")
hs_model.FSM.transitions["toSavingModelAndFeedback"] = fsm.Transition("SavingModelAndFeedback")
hs_model.FSM.transitions["toWipingModelAndFeedback"] = fsm.Transition("WipingModelAndFeedback")
hs_model.FSM.transitions["toDocumentModel"] = fsm.Transition("DocumentModel")
hs_model.FSM.transitions["toLoadModel"] = fsm.Transition("LoadModel")
hs_model.FSM.transitions["toDocumentFeedback"] = fsm.Transition("DocumentFeedback")

hs_model.FSM.Transition("toMenu")
print("About to execute HS_MODEL")
while True:
    hs_model.FSM.Execute()