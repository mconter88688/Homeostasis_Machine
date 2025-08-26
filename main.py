import os # file and directory management
import ctypes

os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libatomic.so.1"

ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("libatomic.so.1", mode=ctypes.RTLD_GLOBAL)

import fsm as fsm
import constants as cons
import models as mod
import states
from allsensors import AllSensors
from disk_storage import get_free_space_gb, is_there_still_space_for_data_collection_and_transfer
import subprocess
import data


### SETUP ###
image_autoencoder = mod.ImageAutoencoder()
image_autoencoder.feature_extractor_setup()

ldrd_autoencoder = mod.LDRD03Autoencoder()

model_data = data.DataCollection()
model_data.load_data(cons.FEEDBACK_FILE)
print("Feedback file loaded")

# Class instance for model parameters
image_model_params = mod.ModelConfigParam(5, 4)
ldrd_model_params = mod.ModelConfigParam(5,4)

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

testing_graphs_path = os.path.join(os.getcwd(), cons.TESTING_GRAPHS_FOLDER)
if not os.path.exists(testing_graphs_path):
    os.makedirs(testing_graphs_path)
print("Testing plots folder exists!")

allsensors = AllSensors()
allsensors.start()


print("About to make HS_MODEL")
hs_model = fsm.HS_Model()
hs_model.FSM.states["NormalDataTraining"] = states.NormalDataTraining(hs_model.FSM, model_data, allsensors, image_autoencoder, ldrd_autoencoder)
hs_model.FSM.states["RLHF"] = states.RLHF(hs_model.FSM, model_data, allsensors, image_autoencoder, ldrd_autoencoder)
hs_model.FSM.states["TrainingImageModel"] = states.TrainingModel(cons.FEEDBACK_FILE, cons.IMAGE_MODEL_PATH, hs_model.FSM, model_data, image_model_params, image_autoencoder, cons.BEST_IMAGE_MODEL_PATH, name_of_model= cons.IMAGE_NAME)
hs_model.FSM.states["TrainingLDRDModel"] = states.TrainingModel(cons.FEEDBACK_FILE, cons.LDRD_MODEL_PATH, hs_model.FSM, model_data,  ldrd_model_params, ldrd_autoencoder, cons.BEST_LDRD_MODEL_PATH, name_of_model= cons.LDRD_NAME)
hs_model.FSM.states["WipingModelAndFeedback"] = states.WipingModelAndFeedback(cons.FEEDBACK_FILE, cons.IMAGE_MODEL_PATH, cons.LDRD_MODEL_PATH, hs_model.FSM, model_data)
hs_model.FSM.states["Menu"] = states.Menu(hs_model.FSM)
hs_model.FSM.states["DocumentImageModel"] = states.DocumentModel(hs_model.FSM, image_model_params, image_autoencoder, cons.IMAGE_MODEL_FOLDER, cons.IMAGE_NAME)
hs_model.FSM.states["DocumentLDRDModel"] = states.DocumentModel(hs_model.FSM, ldrd_model_params, ldrd_autoencoder, cons.LDRD_MODEL_FOLDER, cons.LDRD_NAME)
hs_model.FSM.states["LoadImageModel"] = states.LoadModel(hs_model.FSM, image_model_params, image_autoencoder, cons.IMAGE_MODEL_FOLDER, cons.IMAGE_NAME)
hs_model.FSM.states["LoadLDRDModel"] = states.LoadModel(hs_model.FSM, ldrd_model_params, ldrd_autoencoder, cons.LDRD_MODEL_FOLDER, cons.LDRD_NAME)
hs_model.FSM.states["DocumentFeedback"] = states.DocumentFeedback(hs_model.FSM, [image_model_params, ldrd_model_params], model_data)
hs_model.FSM.states["End"] = states.End(hs_model.FSM, model_data, allsensors)
hs_model.FSM.states["TestingModel"] = states.TestingModel(hs_model.FSM, image_autoencoder, ldrd_autoencoder, model_data, cons.FEEDBACK_FILE)
hs_model.FSM.transitions["toMenu"] = fsm.Transition("Menu")
hs_model.FSM.transitions["toNormalDataTraining"] = fsm.Transition("NormalDataTraining")
hs_model.FSM.transitions["toRLHF"] = fsm.Transition("RLHF")
hs_model.FSM.transitions["toTrainingImageModel"] = fsm.Transition("TrainingImageModel")
hs_model.FSM.transitions["toTrainingLDRDModel"] = fsm.Transition("TrainingLDRDModel")
hs_model.FSM.transitions["toWipingModelAndFeedback"] = fsm.Transition("WipingModelAndFeedback")
hs_model.FSM.transitions["toDocumentImageModel"] = fsm.Transition("DocumentImageModel")
hs_model.FSM.transitions["toDocumentLDRDModel"] = fsm.Transition("DocumentLDRDModel")
hs_model.FSM.transitions["toLoadImageModel"] = fsm.Transition("LoadImageModel")
hs_model.FSM.transitions["toLoadLDRDModel"] = fsm.Transition("LoadLDRDModel")
hs_model.FSM.transitions["toDocumentFeedback"] = fsm.Transition("DocumentFeedback")
hs_model.FSM.transitions["toEnd"] = fsm.Transition("End")
hs_model.FSM.transitions["toTestingModel"] = fsm.Transition("TestingModel")

initial_free = get_free_space_gb(cons.MOUNT_POINT_FOR_STORAGE)
print(f"Starting with {initial_free} GiB free")

hs_model.FSM.Transition("toMenu")
print("About to execute HS_MODEL")
while model_data.program_running:
    if is_there_still_space_for_data_collection_and_transfer(cons.MOUNT_POINT_FOR_STORAGE, initial_free):
        hs_model.FSM.Execute()
    else:
        print(f"Stopping: only {get_free_space_gb(cons.MOUNT_POINT_FOR_STORAGE)} GB free. Shutting down...")
        hs_model.FSM.Transition("toEnd")
        hs_model.FSM.Execute()
        subprocess.run(["sudo", "shutdown", "now"])
