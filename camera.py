from queue import Queue
import numpy as np
import cv2

import pyorbbecsdk as ob
from utils import frame_to_bgr_image


MAX_QUEUE_SIZE = 1
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

class CameraData:
    def __init__(self, ret, frame, processed_frames):
        self.ret = ret
        self.frame = frame
        self.processed_frames = processed_frames

    def return_data_in_list(self):
        return [self.ret, self.frame, self.processed_frames]


class Camera:
    def __init__(self):
        self.pipeline = ob.Pipeline()
        self.device = self.pipeline.get_device()
        self.config = ob.Config()
        self.sensor_list = self.device.get_sensor_list()
        self.video_sensor_types = [
                                        ob.OBSensorType.DEPTH_SENSOR,
                                        ob.OBSensorType.LEFT_IR_SENSOR,
                                        ob.OBSensorType.RIGHT_IR_SENSOR,
                                        ob.OBSensorType.COLOR_SENSOR
                                    ]
        self.hdr_filter = None
        self.frames_queue = Queue()
        # cached frames for better visualization
        self.cached_frames = {
            'color': None,
            'hdr': None,
            'left_ir': None,
            'right_ir': None,
        }
        self.state = None
        self.number= 0

    def configure_streams(self):
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
        
    def configure_HDR(self):
        confighdr = ob.OBHdrConfig()
        confighdr.enable = True
        confighdr.exposure_1 = 7500
        confighdr.gain_1 = 24
        confighdr.exposure_2 = 50
        confighdr.gain_2 = 16
        self.device.set_hdr_config(confighdr)
        self.hdr_filter = ob.HDRMergeFilter()
    
    def start(self):
        try:
            self.pipeline.start(self.config, lambda frames: self.on_new_frame_callback(frames))
        except Exception as e:
            print(e)
            return
        
    def one_capture(self):
        if self.frames_queue.empty():
            return None

        frame_set = self.frames_queue.get()
        if frame_set is None:
            return None

        depth_frame = self.safe_get_depth(frame_set)
        left_ir_frame = self.safe_get_ir(frame_set, ob.OBFrameType.LEFT_IR_FRAME).as_video_frame()
        right_ir_frame = self.safe_get_ir(frame_set, ob.OBFrameType.RIGHT_IR_FRAME).as_video_frame()

        if not all([depth_frame, left_ir_frame, right_ir_frame]):
            return None


        ir_left = np.frombuffer(left_ir_frame.get_data(), dtype=np.uint8).reshape(
            (left_ir_frame.get_height(), left_ir_frame.get_width())
        )
        ir_right = np.frombuffer(right_ir_frame.get_data(), dtype=np.uint8).reshape(
            (right_ir_frame.get_height(), right_ir_frame.get_width())
        )
        color_image = self.process_color(frame_set)
        
        
        if not all([depth_frame, left_ir_frame, right_ir_frame]):
            print("Not All frames received")
            return None
        
        # Process with HDR merge
        merged_frame = self.hdr_filter.process(frame_set)
        if not merged_frame:
            return None
        
        
        merged_frames = merged_frame.as_frame_set()
        merged_depth_frame = merged_frames.get_depth_frame()

        if merged_depth_frame.get_format() == ob.OBFormat.Y16:
            width = merged_depth_frame.get_width()
            height = merged_depth_frame.get_height()
            scale = merged_depth_frame.get_depth_scale()

            merged_depth_data = np.frombuffer(merged_depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
            merged_depth_in_mm = merged_depth_data.astype(np.float32) * scale  # depth in mm
        else:
            raise RuntimeError("merge data not received")


        # Convert frames to displayable images
        merged_depth_image = self.create_depth_image(merged_depth_frame)
        ir_left_image = self.create_ir_image(left_ir_frame)
        ir_right_image = self.create_ir_image(right_ir_frame)

        # Enhance contrast for all images
        # ir_left_image = self.enhance_contrast(ir_left_image, clip_limit=4.0)
        # ir_right_image = self.enhance_contrast(ir_right_image, clip_limit=4.0)
        # merged_depth_image = self.enhance_contrast(merged_depth_image, clip_limit=4.0)

        
        

        # Process all available frames
        processed_frames = {
            'color': color_image,
            'hdr': merged_depth_image,
            'left_ir': ir_left_image,
            'right_ir': ir_right_image
        }

        return_vals = CameraData(True, 
                                 [color_image, merged_depth_in_mm, ir_left, ir_right], 
                                 processed_frames
                                 )
        return return_vals

    
    def process_color(self, frame):
        """Process color frame to BGR image"""
        if not frame:
            return None
        color_frame = frame.get_color_frame()
        color_frame = color_frame if color_frame else self.cached_frames['color']
        if not color_frame:
            return None
        try:
            self.cached_frames['color'] = color_frame
            return frame_to_bgr_image(color_frame)
        except ValueError:
            print("Error processing color frame")
            return None

    def create_depth_image(self, depth_frame):
        """Convert depth frame to colorized image"""
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))
        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)

        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)


    def create_ir_image(self, ir_frame):
        """Convert IR frame to displayable image with enhanced contrast"""
        ir_frame = ir_frame.as_video_frame()
        width = ir_frame.get_width()
        height = ir_frame.get_height()

        ir_data = np.frombuffer(ir_frame.get_data(), dtype=np.uint8)
        ir_data = ir_data.reshape((height, width))

        ir_image = cv2.normalize(ir_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        
        
    def safe_get_depth(self, frame):
        """Process depth frame to colorized depth image"""
        if not frame:
            return None
        depth_frame = frame.get_depth_frame()
        depth_frame = depth_frame if depth_frame else self.cached_frames['hdr']
        if not depth_frame:
            return None
        try:
            self.cached_frames['hdr'] = depth_frame
        except ValueError:
            print("Error processing depth frame")
            return None
        return depth_frame


    def safe_get_ir(self, frame, frame_type):
        if frame is None:
            return None
        ir_frame = frame.get_frame(frame_type)
        frame_name = 'left_ir' if frame_type == ob.OBFrameType.LEFT_IR_FRAME else 'right_ir'
        ir_frame = ir_frame if ir_frame else self.cached_frames[frame_name]
        if not ir_frame:
            return None
        return ir_frame

        
    def on_new_frame_callback(self, frame: ob.FrameSet):
        """Callback function to handle new frames"""
        if frame is None:
            return
        if self.frames_queue.qsize() >= MAX_QUEUE_SIZE:
            self.frames_queue.get()
        self.frames_queue.put(frame)
    
    def create_display(self, processed_frames, width=1280, height=720):
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
        place_frame(processed_frames.get('hdr'), w, 0, width, h)

        # Show stereo IR in bottom row
        place_frame(processed_frames['left_ir'], 0, h, w, height)
        place_frame(processed_frames['right_ir'], w, h, width, height)

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
        add_label("Left IR", 0, h)
        add_label("Right IR", w, h)
        if self.state == "DataIntake":
            add_label(str(self.number), w-15, h*2-60)
        else:
            add_label(self.state, w-25, h*2-60)
        return display
    
    def enhance_contrast(self, image, clip_limit=3.0, tile_grid_size=(8, 8)):
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


    
    
    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
