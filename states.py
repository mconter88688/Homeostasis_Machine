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
from matplotlib.gridspec import GridSpec
import models as mod
import sys
sys.path.append("/home/jon/Homeostasis_machine/rd03_protocol_repo")
from rd03_protocol import RD03Protocol # https://github.com/TimSchimansky/RD-03D-Radar/blob/main/readme.md
import pandas as pd


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
        else:
            cv2.imshow("Control Window", cons.BLANK_SCREEN)
        if all_sensor_data and all_sensor_data.lidar_data and all_sensor_data.rd03_data:
            self.ldrd_temporal_model.all_features_append(all_sensor_data.lidar_data, all_sensor_data.rd03_data)
            if self.ldrd_temporal_model.are_buffers_long_enough():
                self.model_data.append_ld_normal_data(np.stack(self.ldrd_temporal_model.lidar_buffer)) 
                self.model_data.append_rd03_normal_data(np.stack(self.ldrd_temporal_model.radar_buffer))
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.FSM.Transition("toMenu")
        if self.allsensors.gemini:
            self.allsensors.gemini.number+=1


    def Exit(self):
        self.temporal_model.buffer.clear()
        cv2.destroyAllWindows()

class WipingModelAndFeedback(fsm.State):
    def __init__(self, FEEDBACK_FILE, IMAGE_MODEL_PATH, LDRD_MODEL_PATH, FSM, model_data):
        self.FEEDBACK_FILE = FEEDBACK_FILE
        self.IMAGE_MODEL_PATH = IMAGE_MODEL_PATH
        self.LDRD_MODEL_PATH = LDRD_MODEL_PATH
        self.FSM = FSM
        self.model_data = model_data

    def Enter(self):
        print("Deleting model and feedback data")
    
    def Execute(self):
        self.model_data.clear_data()
        
        # Clear Feedback Data
        with open(self.FEEDBACK_FILE, "wb") as f:
            pass

        if os.path.exists(self.IMAGE_MODEL_PATH):
            os.remove(self.IMAGE_MODEL_PATH)
        if os.path.exists(self.LDRD_MODEL_PATH):
             os.remove(self.LDRD_MODEL_PATH)
        print("Feedback and model successfully removed!")
        self.FSM.Transition("toMenu")
        return
    
        
    

    
class Menu(fsm.State):
    def __init__(self, FSM):
        self.FSM = FSM
    
    def Enter(self):
        print("**Select from the following:**")
        print("Wipe Model and Feedback:...........................W")
        print("Retrain Model:.....................................R")
        print("Take in Training Data:.............................N")
        print("Test Your Data.....................................T")
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
            answer = input("Would you like to train the image (I) or the LDRD (L) model?").strip().upper()
            if answer == "I":
                self.FSM.Transition("toTrainingImageModel")
            elif answer == "L":
                self.FSM.Transition("toTrainingLDRDModel")
            else:
                print("Invalid Input")
        elif answer == "N":
            self.FSM.Transition("toNormalDataTraining")
        elif answer == "T":
            self.FSM.Transition("toTestingModel")
        elif answer == "F":
            self.FSM.Transition("toRLHF")
        elif answer == "M":
            answer = input("Would you like to document the image (I) or the LDRD (L) model?").strip().upper()
            if answer == "I":
                self.FSM.Transition("toDocumentImageModel")
            elif answer == "L":
                self.FSM.Transition("toDocumentLDRDModel")
        elif answer == "L":
            answer = input("Would you like to load in the image (I) or the LDRD (L) model?").strip().upper()
            if answer == "I":
                self.FSM.Transition("toLoadImageModel")
            elif answer == "L":
                self.FSM.Transition("toLoadLDRDModel")
        elif answer == "D":
            self.FSM.Transition("toDocumentFeedback")
        elif answer == "Q":
            self.FSM.Transition("toEnd")
        else:
            print("Invalid input. Try again.")


        



class TrainingModel(fsm.State):
    def __init__(self, FEEDBACK_FILE, MODEL_PATH, FSM, model_data, model_params, temporal_model, BEST_MODEL_PATH, name_of_model = ""):
        self.FEEDBACK_FILE = FEEDBACK_FILE
        self.MODEL_PATH = MODEL_PATH
        self.FSM = FSM
        self.model_data = model_data
        self.model_params = model_params
        self.temporal_model = temporal_model
        self.BEST_MODEL_PATH = BEST_MODEL_PATH
        self.name_of_model = name_of_model
    
    def Enter(self):
        print("Training " + self.name_of_model + " Model")

    def Execute(self):
        # Retrain model
        self.model_params.callbacks = [
                        EarlyStopping(patience=3, restore_best_weights=True),
                        ModelCheckpoint(self.BEST_MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
                    ]
        #print(self.model_params.callbacks)
        epoch_num = int(input("Epochs: "))
        batch_num = int(input("Batch Size: "))
        validation_num = float(input("Validation Split: "))
        self.model_params.redefine_all(epoch_num, batch_num, validation_num, None, None)
        answer = input("Would you like to load a saved data file?").strip().upper()
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
        self.model_data.save_data(self.FEEDBACK_FILE)
        
        if not self.model_data.is_empty():
            print("Retraining model with feedback data...")
            # Create training sets
            if self.name_of_model == cons.IMAGE_NAME:
                X = np.array(self.model_data.normal_data)
            elif self.name_of_model == cons.LDRD_NAME:
                X = [np.array(self.model_data.ld_normal_data), np.array(self.model_data.rd03_normal_data)]
                #print(X)
            else: 
                print("Invalid model type. Returning to main menu.")
                self.FSM.Transition("toMenu")
                return
            # y = np.array([0]*len(self.model_data.normal_data) + [1]*len(self.model_data.anomaly_data)) # trains it with predictions being certain of normal v.s. anomaly scenarios
            history = self.temporal_model.fit(self.model_params, X,X)
            self.temporal_model.model.save(self.MODEL_PATH)
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
                    
                    if self.name_of_model == cons.IMAGE_NAME:
                        self.FSM.Transition("toDocumentImageModel")
                        return
                    elif self.name_of_model == cons.LDRD_NAME:
                        self.FSM.Transition("toDocumentLDRDModel")
                        return
                    else: 
                        print("Model type not detected, so not saved")
                        self.FSM.Transition("toMenu")
                        return
                    return

        self.FSM.Transition("toMenu")
        return



class LoadModel(fsm.State):
    def __init__(self, FSM, model_params, temporal_model, MODEL_FOLDER, name_of_model):
        self.FSM = FSM
        self.model_params = model_params
        self.temporal_model = temporal_model
        self.MODEL_FOLDER = MODEL_FOLDER
        self.name_of_model = name_of_model

    def Enter(self):
        print("Available model folders:")
        for folder in os.listdir(self.MODEL_FOLDER):
            print("-", folder)

    def Execute(self):
        if not os.listdir(self.MODEL_FOLDER):
            print(self.name_of_model + " MODEL_FOLDER is empty.")
            self.FSM.Transition("toMenu")
            return
        good_model = False
        while not good_model:
            answer = input("Select " + self.name_of_model + " model to load: ").replace(" ", "")
            if answer.upper() == "Q":
                print("Quitting...")
                self.FSM.Transition("toMenu")
                return
            if answer in os.listdir(self.MODEL_FOLDER):
                good_model = True
                self.temporal_model.model = load_model(os.path.join(os.getcwd(), self.MODEL_FOLDER, answer, answer + ".h5"))
                self.model_params.epochs = 0
                self.model_params.batch_size = 0
                self.model_params.validation_split = 0
                self.model_params.model_file = answer + ".h5"
                self.model_params.feedback_file = None
            else:
                print("Model does not exist. Try again")
        self.FSM.Transition("toMenu")



class DocumentFeedback(fsm.State):
    def __init__(self, FSM, model_params_list, model_data):
        self.FSM = FSM
        self.model_data = model_data
        self.model_params_list = model_params_list
    
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
        for model_params in self.model_params_list:
            model_params.feedback_file = file_path
        notes = input("Notes: ")
        with open(info_path, 'w') as f:
            f.write(notes)
        self.FSM.Transition("toMenu")
        pass



class DocumentModel(fsm.State):
    def __init__(self, FSM, model_params, temporal_model, MODEL_FOLDER, name_of_model):
        self.FSM = FSM
        self.model_params = model_params
        self.temporal_model = temporal_model
        self.MODEL_FOLDER = MODEL_FOLDER
        self.name_of_model = name_of_model

    def Enter(self):
        print("Documenting the " + self.name_of_model + " model")

    def Execute(self):
        good_file = False
        while not good_file:
            answer = input("Name of model: ").replace(" ", "")
            if answer.upper() == "Q":
                print("Quitting...")
                self.FSM.Transition("toMenu")
                return
            file_name = answer + ".h5"
            folder_path = os.path.join(os.getcwd(), self.MODEL_FOLDER, answer)
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

class TestingModel(fsm.State):
    def __init__(self, FSM, temporal_model, ldrd_temporal_model, model_data, FEEDBACK_FILE):
        self.FSM = FSM
        self.temporal_model = temporal_model
        self.ldrd_temporal_model = ldrd_temporal_model
        self.model_data = model_data
        self.FEEDBACK_FILE = FEEDBACK_FILE

    def Enter(self):
        print("Model Testing")

    def Execute(self):
        answer = input("Would you like to load a saved data file?").strip().upper()
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
        self.model_data.save_data(self.FEEDBACK_FILE)
        
        test_data_length = len(self.model_data.normal_data)
        ldrd_predictions = np.zeros(test_data_length)
        image_predictions = np.zeros(test_data_length)
        for i in range(test_data_length):
            image_predictions[i] = self.temporal_model.predict(self.model_data.normal_data[i])
            ldrd_predictions[i] = self.ldrd_temporal_model.predict(self.model_data.ld_normal_data[i], self.model_data.rd03_normal_data[i])
        total_predictions = mod.late_fusion(image_predictions, ldrd_predictions)
        data = [total_predictions, image_predictions, ldrd_predictions]
        labels = ["Weighted Average\nReconstruction\nError",
                    "Image Autoencoder\nAverage Reconstruction\nError",
                    "LiDAR and MMWave\nAutoencoder Average Reconstruction\nError"]
        
        
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 2, width_ratios=[1.5, 1])  # left column wider

        # ---- Left panel: combined view ----
        ax_left = fig.add_subplot(gs[:, 0])  # span all rows
        ax_left.boxplot(data, patch_artist=True, labels=labels)
        ax_left.set_ylabel("Average Reconstruction Error")
        ax_left.set_title("All Predictions Together")

        # ---- Right panel: 3 separate boxplots ----
        for i, (preds, label) in enumerate(zip(data, labels)):
            ax = fig.add_subplot(gs[i, 1])  # one subplot per row
            ax.boxplot(preds, patch_artist=True, labels=[label])
            ax.set_ylabel("Average Reconstruction Error")

            # Compute summary stats
            q1 = np.percentile(preds, 25)
            median = np.percentile(preds, 50)
            q3 = np.percentile(preds, 75)
            min_val = preds.min()
            max_val = preds.max()

            stats_text = (
                f"Min={min_val:.3f}, Q1={q1:.3f}, "
                f"Median={median:.3f}, Q3={q3:.3f}, Max={max_val:.3f}"
            )

            # Place stats under each box
            ax.text(
                1,
                ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                stats_text,
                ha="center", va="top", fontsize=8
            )

            ax.margins(y=0.2)

        # ---- Global title ----
        answer = input("Name of Graph: ")
        fig.suptitle(answer, fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 

        graph_path = os.path.join(os.getcwd(), "temp_testing_plot.png")
        plt.savefig(graph_path)

        img = cv2.imread(graph_path)
        if img is not None:
            cv2.imshow("Graph", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        answer = input("Would you like to save the plot and data? (Y/N)").strip().upper()
        if answer == "Y":
            if os.path.exists(graph_path):
                name_of_graph = input("Name of plot: ")
                graph_target = os.path.join(os.getcwd(), cons.TESTING_GRAPHS_FOLDER, name_of_graph + ".png")
                os.rename(graph_path, graph_target)
                df = pd.DataFrame({
                    "Total": total_predictions,
                    "Image": image_predictions,
                    "LDRD": ldrd_predictions
                })
                csv_target = os.path.join(os.getcwd(), cons.TESTING_GRAPHS_FOLDER, name_of_graph + ".csv")
                df.to_csv(csv_target, index=False)
                print("Plot and data saved!")
            else:
                print("Something went wrong, plot was not saved.")
        else:
            print("You did not request  the plot to be saved. Going to menu...")
        self.FSM.Transition("toMenu")



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
            self.ldrd_temporal_model.all_features_append(all_sensor_data.lidar_data, all_sensor_data.rd03_data)
           
            if self.ldrd_temporal_model.are_buffers_long_enough():
                pred_ldrd = self.ldrd_temporal_model.model_prediction()  

        if pred_image and pred_ldrd:
            pred = mod.late_fusion(pred_image, pred_ldrd)
        elif pred_image:
            pred = pred_image
        elif pred_ldrd:
            pred = pred_ldrd

        if pred:    
            is_anomaly = pred > cons.ANOMALY_THRESHOLD #checks prediction against threshold
            self.allsensors.gemini.state = f"{'ANOMALY' if is_anomaly else 'NORMAL'} ({pred:.2f})"

            # Draw on frame
        if all_sensor_data:
            display = self.allsensors.gemini.create_display(all_sensor_data.camera_data.processed_frames)
            cv2.imshow("Feedback Data", display)
        else:
            cv2.imshow("Control Window", cons.BLANK_SCREEN)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.FSM.Transition("toMenu")
            return
        # elif key == ord('n') and self.temporal_model.is_buffer_long_enough():
        #     self.model_data.append_normal_data(np.stack(self.buffer))
        #     print("Labeled one normal sequence")
        # elif key == ord('a') and self.temporal_model.is_buffer_long_enough():
        #     self.model_data.append_anomaly_data(np.stack(self.buffer))
        #     print("Labeled one anomalous sequence")

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