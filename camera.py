from queue import Queue
import cv2
import numpy as np
import time

import pyorbbecsdk as ob
from utils import frame_to_bgr_image


MAX_QUEUE_SIZE = 1

class Camera:
    def __init__(self):
        self.pipeline = ob.Pipeline()
        self.device = self.Pipeline.get_device()
        self.config = ob.Config()
        self.sensor_list = self.device.get_sensor_list()
        self.video_sensor_types = [
                                        ob.OBSensorType.DEPTH_SENSOR,
                                        ob.OBSensorType.LEFT_IR_SENSOR,
                                        ob.OBSensorType.RIGHT_IR_SENSOR,
                                        ob.OBSensorType.IR_SENSOR,
                                        ob.OBSensorType.COLOR_SENSOR
                                    ]
        self.hdr_filter = None
        self.frames_queue = ob.Queue()

    def ConfigureStreams(self):
        # enable all wanted streams
        for sensor in range(len(self.sensor_list)):
            sensor_type = self.sensor_list[sensor].get_type()
            if sensor_type in self.video_sensor_types:
                try:
                    print(f"Enabling sensor type: {sensor_type}")
                    self.config.enable_stream(sensor_type)
                except:
                    print(f"Failed to enable sensor type: {sensor_type}")
                    continue

        
        # configure streams
        self.config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE) # requires all frames to be from same timestamp
        try:
            self.pipeline.enable_frame_sync() # hardware-level synchronization
        except Exception as e:
            print(e)
        
    def ConfigureHDR(self):
        confighdr = ob.OBHdrConfig()
        confighdr.enable = True
        confighdr.exposure_1 = 7500
        confighdr.gain_1 = 24
        confighdr.exposure_2 = 50
        confighdr.gain_2 = 16
        self.device.set_hdr_config(confighdr)
        self.hdr_filter = ob.HDRMergeFilter()
    
    def Start(self):
        self.pipeline.enable_frame_sync() # sync all sensor frames
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(e)
            return
        


    def Capture(self):
        pass
        
    def on_new_frame_callback(self, frame: ob.FrameSet):
        """Callback function to handle new frames"""
        if frame is None:
            return
        if self.frames_queue.qsize() >= MAX_QUEUE_SIZE:
            self.frames_queue.get()
        self.frames_queue.put(frame)
    
    def create_display(processed_frames, width=1280, height=720):
        """Create display window with all processed frames
        Layout:
        2x2 grid :
        [Color] [HR]
        [L-IR] [R-IR]
        """
        display = np.zeros((height, width, 3), dtype=np.uint8)
        h, w = height // 2, width // 2

        # Helper function for safe image resizing
        def safe_resize(img, target_size):
            if img is None:
                return None
            try:
                return cv2.resize(img, target_size)
            except:
                return None

        # Process frames with consistent error handling
        def place_frame(img, x1, y1, x2, y2):
            if img is not None:
                try:
                    h_section = y2 - y1
                    w_section = x2 - x1
                    resized = safe_resize(img, (w_section, h_section))
                    if resized is not None:
                        display[y1:y2, x1:x2] = resized
                except:
                    pass

        # Always show color and depth in top row if available
        place_frame(processed_frames.get('color'), 0, 0, w, h)
        place_frame(processed_frames.get('depth'), w, 0, width, h)

        # Handle IR display in bottom row
        has_left_ir = processed_frames.get('left_ir') is not None
        has_right_ir = processed_frames.get('right_ir') is not None
        has_single_ir = processed_frames.get('ir') is not None

        if has_left_ir and has_right_ir:
            # Show stereo IR in bottom row
            place_frame(processed_frames['left_ir'], 0, h, w, height)
            place_frame(processed_frames['right_ir'], w, h, width, height)
        elif has_single_ir:
            # Show single IR in bottom-left quadrant
            place_frame(processed_frames['ir'], 0, h, w, height)

        # Add labels to identify each stream
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)
        font_thickness = 2

        # Helper function for adding labels
        def add_label(text, x, y):
            cv2.putText(display, text, (x + 10, y + 30), font, font_scale,
                    font_color, font_thickness)

        # Add labels for each quadrant
        add_label("Color", 0, 0)
        add_label("HDR", w, 0)

        if has_left_ir and has_right_ir:
            add_label("Left IR", 0, h)
            add_label("Right IR", w, h)
        elif has_single_ir:
            add_label("IR", 0, h)

        return display
    
    def EnhanceContrast(self, image, clip_limit=3.0, tile_grid_size=(8, 8)):
        """
        Enhance image contrast using CLAHE
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)
        
    
        
            


    
    
    def Stop(self):
        self.Pipeline.stop()