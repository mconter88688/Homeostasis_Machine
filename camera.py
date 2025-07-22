import cv2
import numpy as np
import time

import pyorbbecsdk as ob
from utils import frame_to_bgr_image



class Camera:
    def __init__(self):
        self.Pipeline = ob.Pipeline()

    def Start(self):
        self.Pipeline.start()

    def Capture(self, wait_time):
        frames = self.Pipeline.wait_for_frames(wait_time)
        if frames is None:
            return
        
            


    
    
    def Stop(self):
        self.Pipeline.stop()