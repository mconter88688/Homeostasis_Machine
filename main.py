import fsm as fsm
print("fsm")
from tensorflow.keras.models import load_model # for model architecture and loading
print("tf.k.mod")
from collections import deque # for sliding window
print("deque")
import os # file and directory management
print("os")
import pickle
print("pickle")
import constants as cons
print("cons")
import models as mod
print("mod")
#from sklearn.model_selection import train_test_split
import camera as cam
print("camera")
import LiDAR as ld
print("lidar")
import states
print("states")
from rd03_protocol import RD03Protocol # https://github.com/TimSchimansky/RD-03D-Radar/blob/main/readme.md

class Data:
    def __init__(self, feature_extractor):
        self.normal_data = []
        self.anomaly_data = []
        self.feature_extractor = feature_extractor
    
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
program_running = True
buffer = deque(maxlen=cons.SEQ_LEN)

feature_extractor = mod.feature_extractor_setup()
# Load previous feedback data if it exists
model_data = Data(feature_extractor)
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



if os.path.exists(cons.MODEL_PATH):
    temporal_model = load_model(cons.MODEL_PATH) # function imported from tensorflow.keras.models
else:
    temporal_model = mod.build_model()

camera = cam.Camera()
camera.configure_streams()
camera.configure_HDR()
camera.start()

ld19 = ld.LD19()
ld19.start()

radar = RD03Protocol("/dev/ttyUSB1", enable_plot=False)


print("About to make HS_MODEL")
hs_model = fsm.HS_Model()
hs_model.FSM.states["NormalDataTraining"] = states.NormalDataTraining(hs_model.FSM, model_data, camera, ld19, buffer, radar)
hs_model.FSM.states["RLHF"] = states.RLHF(hs_model.FSM, model_data, camera, ld19, buffer, temporal_model, radar)
hs_model.FSM.states["SavingModelAndFeedback"] = states.SavingModelAndFeedback(cons.FEEDBACK_FILE, cons.MODEL_PATH, hs_model.FSM, model_data, model_params, temporal_model)
hs_model.FSM.states["WipingModelAndFeedback"] = states.WipingModelAndFeedback(cons.FEEDBACK_FILE, cons.MODEL_PATH, hs_model.FSM, model_data)
hs_model.FSM.states["Menu"] = states.Menu(hs_model.FSM)
hs_model.FSM.states["DocumentModel"] = states.DocumentModel(hs_model.FSM, model_params, temporal_model)
hs_model.FSM.states["LoadModel"] = states.LoadModel(hs_model.FSM, model_params, temporal_model)
hs_model.FSM.states["DocumentFeedback"] = states.DocumentFeedback(hs_model.FSM, model_params, model_data)
hs_model.FSM.states["End"] = states.End(hs_model.FSM, program_running, radar, ld19, camera)
hs_model.FSM.transitions["toMenu"] = fsm.Transition("Menu")
hs_model.FSM.transitions["toNormalDataTraining"] = fsm.Transition("NormalDataTraining")
hs_model.FSM.transitions["toRLHF"] = fsm.Transition("RLHF")
hs_model.FSM.transitions["toSavingModelAndFeedback"] = fsm.Transition("SavingModelAndFeedback")
hs_model.FSM.transitions["toWipingModelAndFeedback"] = fsm.Transition("WipingModelAndFeedback")
hs_model.FSM.transitions["toDocumentModel"] = fsm.Transition("DocumentModel")
hs_model.FSM.transitions["toLoadModel"] = fsm.Transition("LoadModel")
hs_model.FSM.transitions["toDocumentFeedback"] = fsm.Transition("DocumentFeedback")
hs_model.FSM.transitions["toEnd"] = fsm.Transition("End")

hs_model.FSM.Transition("toMenu")
print("About to execute HS_MODEL")
while program_running:
    hs_model.FSM.Execute()