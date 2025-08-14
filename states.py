import fsm
from time import sleep
import LiDAR as ld
import constants as cons
import numpy as np # for arrays
import cv2 # for camera
import os # file and directory management
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model # for model architecture and loading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import models as mod
import sys
sys.path.append("/home/jon/Homeostasis_machine/rd03_protocol_repo")
from rd03_protocol import RD03Protocol # https://github.com/TimSchimansky/RD-03D-Radar/blob/main/readme.md

class NormalDataTraining(fsm.State):
    def __init__(self, FSM, model_data, allsensors, temporal_model, ldrd_temporal_model):
        self.FSM = FSM
        self.num_frames = 0
        self.model_data = model_data
        self.allsensors = allsensors
        self.temporal_model = temporal_model
        self.ldrd_temporal_model = ldrd_temporal_model

    def Enter(self):
        print("Normal Feedback Data Mode")
        if self.allsensors.gemini:
            self.allsensors.gemini.number = 0
            self.allsensors.gemini.state = "NormalDataTraining"

    
    def Execute(self):
        # Train on only normal feedback
        all_sensor_data = self.allsensors.capture_sensor_info()
        if all_sensor_data and all_sensor_data.camera_data:
            feat = self.temporal_model.feature_extract_combine(all_sensor_data.camera_data.frame)
            self.temporal_model.feature_append(feat)
            if self.temporal_model.is_buffer_long_enough():
                self.model_data.append_normal_data(np.stack(self.temporal_model.buffer))
            # Create and display the combined view
            display = self.allsensors.gemini.create_display(all_sensor_data.camera_data.processed_frames)
            cv2.imshow("Normal data", display)
        if all_sensor_data and all_sensor_data.lidar_data and all_sensor_data.rd03_data:
            self.ldrd_temporal_model.all_features_append(all_sensor_data.lidar_data, all_sensor_data.rd03_data)
            if self.ldrd_temporal_model.are_buffers_long_enough():
                self.model_data.append_ldrd_normal_data([np.stack(self.ldrd_temporal_model.lidar_buffer), 
                                                         np.stack(self.ldrd_temporal_model.radar_buffer)])
        else:
            cv2.imshow("Control Window", cons.BLANK_SCREEN)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.FSM.Transition("toMenu")
        if self.allsensors.gemini:
            self.allsensors.gemini.number+=1


    def Exit(self):
        self.temporal_model.buffer.clear()
        cv2.destroyAllWindows()

class WipingModelAndFeedback(fsm.State):
    def __init__(self, FEEDBACK_FILE, IMAGE_MODEL_PATH, FSM, model_data):
        self.FEEDBACK_FILE = FEEDBACK_FILE
        self.IMAGE_MODEL_PATH = IMAGE_MODEL_PATH
        self.FSM = FSM
        self.model_data = model_data

    def Enter(self):
        print("Deleting model and feedback data")
    
    def Execute(self):
        self.model_data.clear_data()
        
        with open(cons.FEEDBACK_FILE, "wb") as f:
            pass

        if os.path.exists(self.IMAGE_MODEL_PATH):
            os.remove(self.IMAGE_MODEL_PATH)
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
        print("Exit...............................................Q")

            
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
        elif answer == "Q":
            self.FSM.Transition("toEnd")
        else:
            print("Invalid input. Try again.")

    def Exit(self):
        pass


class SavingModelAndFeedback(fsm.State):
    def __init__(self, FEEDBACK_FILE, IMAGE_MODEL_PATH, FSM, model_data, model_params, temporal_model):
        self.FEEDBACK_FILE = FEEDBACK_FILE
        self.IMAGE_MODEL_PATH = IMAGE_MODEL_PATH
        self.FSM = FSM
        self.model_data = model_data
        self.model_params = model_params
        self.temporal_model = temporal_model
    
    def Enter(self):
        print("Saving Model and Feedback File")

    def Execute(self):
        # Retrain model
        self.model_params.callbacks = [
                        EarlyStopping(patience=3, restore_best_weights=True),
                        ModelCheckpoint(cons.BEST_IMAGE_MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
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
            history = self.temporal_model.fit(self.model_params, X,X)
            self.temporal_model.model.save(cons.IMAGE_MODEL_PATH)
            print("Model updated and saved.")

            answer = input("Would you like to graph the data? (Y/N)").strip().upper()
            if answer == "Y":
                # Plot training & validation loss and accuracy
                plt.figure(figsize=(10, 4))

                #plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.title('Loss Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                # plt.subplot(1, 2, 2)
                # plt.plot(history.history['accuracy'], label='Train Acc')
                # plt.plot(history.history['val_accuracy'], label='Val Acc')
                # plt.title('Accuracy Over Epochs')
                # plt.xlabel('Epoch')
                # plt.ylabel('Accuracy')
                # plt.legend()

                plt.tight_layout()

                graph_path = os.path.join(os.getcwd(), "temp_training_plot.png")
                plt.savefig(graph_path)

                img = cv2.imread(graph_path)
                if img is not None:
                    cv2.imshow("Graph", img)
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
                self.temporal_model.model = load_model(os.path.join(os.getcwd(), cons.MODEL_FOLDER, answer, answer + ".h5"))
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
        self.temporal_model.model.save(file_path)
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
    def __init__(self, FSM, model_data, allsensors, temporal_model, ldrd_temporal_model):
        self.FSM = FSM
        self.model_data = model_data
        self.allsensors = allsensors
        self.temporal_model = temporal_model
        self.ldrd_temporal_model = ldrd_temporal_model

    def Enter(self):
        print("Human Feedback Mode")
        print("Press 'n' to label homeostasis, 'a' to label abnormalities, and 'q' to quit.")
        if self.allsensors.gemini:
            self.allsensors.gemini.state = "RLHF"

    def Execute(self):
        pred = None
        pred_image = None
        pred_ldrd = None
        all_sensor_data = self.allsensors.capture_sensor_info()
        if all_sensor_data and all_sensor_data.camera_data:
            feat = self.temporal_model.feature_extract_combine(all_sensor_data.camera_data.frame)
            self.temporal_model.feature_append(feat) # add 1D array to end of the buffer

            if self.temporal_model.is_buffer_long_enough():
                pred_image = self.temporal_model.model_prediction()
        
        if all_sensor_data and all_sensor_data.lidar_data and all_sensor_data.rd03_data:
            self.ldrd_temporal_model.all_features_append(feat)
           
            if self.ldrd_temporal_model.are_buffers_long_enough():
                pred_ldrd = self.ldrd_temporal_model.model_prediction()  

        if pred_image and pred_ldrd:
            pred = cons.IMAGE_COEFF*pred_image + (1-cons.IMAGE_COEFF)*pred_ldrd
        elif pred_image:
            pred = pred_image
        elif pred_ldrd:
            pred = pred_ldrd

        if pred:    
            is_anomaly = pred > cons.ANOMALY_THRESHOLD #checks prediction against threshold
            self.allsensors.gemini.state = f"{'ANOMALY' if is_anomaly else 'NORMAL'} ({pred:.2f})"

            # Draw on frame
        display = self.allsensors.gemini.create_display(all_sensor_data.camera_data.processed_frames)
        cv2.imshow("Feedback Data", display)
        # else:
        #     cv2.imshow("Control Window", cons.BLANK_SCREEN)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.FSM.Transition("toMenu")
            return
        elif key == ord('n') and self.temporal_model.is_buffer_long_enough():
            self.model_data.append_normal_data(np.stack(self.buffer))
            print("Labeled one normal sequence")
        elif key == ord('a') and self.temporal_model.is_buffer_long_enough():
            self.model_data.append_anomaly_data(np.stack(self.buffer))
            print("Labeled one anomalous sequence")

    def Exit(self):
        self.temporal_model.buffer.clear()
        cv2.destroyAllWindows()

class End(fsm.State):
    def __init__(self, FSM, model_data, allsensors):
        self.FSM = FSM
        self.model_data = model_data
        self.allsensors = allsensors

    def Enter(self):
        print("Ending...")
        self.allsensors.stop()
        cv2.destroyAllWindows()
        self.model_data.program_running = False

    def Execute(self):
        pass

    def End(self):
        pass