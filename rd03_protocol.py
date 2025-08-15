import serial
from dataclasses import dataclass
from typing import Optional
import constants as cons
from sensor import Sensor
import numpy as np
from dsp import ema, within_radius
from time import monotonic_ns

RADAR_EMA_ALPHA = 0.6
RADAR_MAX_RANGE = 8000
RADAR_MIN_RANGE = 10



class RadarPreprocessedData:
    def __init__(self):
        self.x_coords = np.zeros(cons.RADAR_MAX_TARGETS) # rewritten in update_for_new_target_data
        self.y_coords = np.zeros(cons.RADAR_MAX_TARGETS) # rewritten in update_for_new_target_data
        self.speeds = np.zeros(cons.RADAR_MAX_TARGETS) # rewritten in update_for_new_target_data
        self.distances = np.zeros(cons.RADAR_MAX_TARGETS) # rewritten in update_for_new_target_data
        self.timestamp = None # rewritten in update_for_new_target_data
        self.are_there_prev_vals = False # rewritten in ema
        self.prev_x_coords = np.zeros(cons.RADAR_MAX_TARGETS) # rewritten  in ema
        self.prev_y_coords = np.zeros(cons.RADAR_MAX_TARGETS) # rewritten in ema
        self.prev_speeds = np.zeros(cons.RADAR_MAX_TARGETS) # rewritten  in ema
        self.prev_distances = np.zeros(cons.RADAR_MAX_TARGETS) # rewritten in ema

    def update_for_new_target_data(self, radar_target):
        self.timestamp = monotonic_ns()
        for i in range(cons.RADAR_MAX_TARGETS):
            self.x_coords[i] = radar_target[i].x_coord
            self.y_coords[i] = radar_target[i].y_coord
            self.speeds[i] = radar_target[i].speed
            self.distances[i] = radar_target[i].distance

    def class_to_single_numpy_array(self):
        return np.column_stack((self.x_coords, self.y_coords,  self.speeds, self.distances))

    def copy(self):
        new_obj = RadarPreprocessedData()
        new_obj.are_there_prev_vals = self.are_there_prev_vals
        new_obj.timestamp = self.timestamp
        new_obj.x_coords = self.x_coords.copy()
        new_obj.y_coords = self.y_coords.copy()
        new_obj.speeds = self.speeds.copy()
        new_obj.distances = self.distances.copy()
        new_obj.prev_x_coords = self.prev_x_coords.copy()
        new_obj.prev_y_coords = self.prev_y_coords.copy()
        new_obj.prev_speeds = self.prev_speeds.copy()
        new_obj.prev_distances = self.prev_distances.copy()
        return new_obj


    def ema(self):
        self.x_coords[:] = ema(self.prev_x_coords, self.x_coords, self.are_there_prev_vals, alpha = RADAR_EMA_ALPHA)
        self.y_coords[:] = ema(self.prev_y_coords, self.y_coords, self.are_there_prev_vals, alpha = RADAR_EMA_ALPHA)
        self.speeds[:] = ema(self.prev_speeds, self.speeds, self.are_there_prev_vals, alpha = RADAR_EMA_ALPHA)
        self.distances[:] = ema(self.prev_distances, self.distances, self.are_there_prev_vals, alpha = RADAR_EMA_ALPHA)
        self.are_there_prev_vals = True
        
        



@dataclass
class RadarTarget:
    """Represents a single radar target's data"""
    x_coord: float      # mm, positive or negative
    y_coord: float      # mm, positive or negative
    speed: float        # cm/s, positive or negative
    distance: float     # mm, pixel distance value

    def copy(self):
        targets = []
        for target in self.latest_data:
            #print("scan found target")
            targets.append(RadarTarget(
                x_coord=target.x_coord,
                y_coord=target.y_coord,
                speed=target.speed,
                distance=target.distance  
            ))
        return targets


class RD03Protocol(Sensor):
    HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
    FOOTER = bytes([0x55, 0xCC])
    TARGET_DATA_SIZE = 8
    
    WAITING_HEADER = 0
    READING_DATA = 1
    WAITING_FOOTER = 2
    
    # Number of positions to keep in trace history
    TRACE_LENGTH = 20

    def __init__(self):
        """Initialize the RD03D Protocol handler with serial port settings"""
        super().__init__(name = "RD03", baudrate=256000, port=cons.RD03D_PORT)

   
    def start(self):
        super().start(bytesize = serial.EIGHTBITS, 
                      parity = serial.PARITY_NONE, 
                      stopbits = serial.STOPBITS_ONE, 
                      timeout = cons.TIMEOUT
                      )
        
    
    def _decode_raw(self, value: int) -> float:
        """Decode a coordinate value according to the protocol specification"""
        # Check if highest bit is set (positive/negative indicator)
        is_negative = not bool(value & 0x8000)
        # Get absolute value (15 bits)
        abs_value = value & 0x7FFF
        return -abs_value if is_negative else abs_value

    def dsp_and_send_scan(self, radar_target, radar_preprocessed_data):
        radar_preprocessed_data.update_for_new_target_data(radar_target)
        radar_preprocessed_data.ema()
        with self.lock:
            self.latest_data = radar_preprocessed_data    

    
    def _parse_target_data(self, data: bytes) -> Optional[RadarTarget]:
        """Parse 8 bytes of target data into a RadarTarget object"""
        # if all(b == 0 for b in data):  # Check if target data is all zeros
        #     return None
            
        # Extract values (little endian)
        x_raw = int.from_bytes(data[0:2], byteorder='little')
        y_raw = int.from_bytes(data[2:4], byteorder='little')
        speed_raw = int.from_bytes(data[4:6], byteorder='little')
        distance = int.from_bytes(data[6:8], byteorder='little')

        return RadarTarget(
            x_coord=self._decode_raw(x_raw),
            y_coord=self._decode_raw(y_raw),
            speed=self._decode_raw(speed_raw),
            # TODO: I dont get what this does and also this should be uin16?!
            distance=float(distance)  
        )
        


    def _reader_thread(self):
        """Read and parse a complete data frame from the radar"""
        print("read_frame running")
        frame_data = bytearray()
        header_found = False
        radar_preprocessed_data = RadarPreprocessedData()

        while self.running:
            within_distance = True
            if self.serial.in_waiting:
                byte = ord(self.serial.read())
                #print(hex(byte))
                #print("read rd03d serial")
                
                if not header_found:
                    # print("no header found")
                    # print(byte)
                    frame_data.append(byte)
                    # Check for header sequence
                    if len(frame_data) >= 4:
                        if (frame_data[-4:] == bytes([0xAA, 0xFF, 0x03, 0x00])):
                            header_found = True
                            frame_data = frame_data[-4:]  # Keep only the header
                elif header_found:
                    # print("header found")
                    frame_data.append(byte)
                    
                    # Check if we have a complete frame
                    if len(frame_data) >= (4 + 24 + 2):  # Header + 3*8 bytes data + Footer
                        #print("complete frame")
                        if frame_data[-2:] == bytes([0x55, 0xCC]):
                            # Valid frame received, parse targets
                            #print("valid frame: " + str(len(frame_data)))
                            targets = []
                            data_start = 4  # After header
                            
                            for i in range(3):  # 3 possible targets
                                target_data = frame_data[data_start + i*8:data_start + (i+1)*8]
                                target = self._parse_target_data(target_data)
                                if target is not None:
                                    if not within_radius(target.distance, RADAR_MIN_RANGE, RADAR_MAX_RANGE):
                                        within_distance = False
                                        continue
                                    
                                    # print("target " + str(i) + " found")
                                    targets.append(target)
                            # for target in targets:
                            #     print(f"Target at ({target.x_coord}, {target.y_coord}), Speed: {target.speed}")
                            
                            frame_data = bytearray()
                            header_found = False
                            if within_distance:
                                self.dsp_and_send_scan(targets, radar_preprocessed_data)
                            
                        else:
                            print("invalid frame")
                            # Invalid frame, start over
                            frame_data = bytearray()
                            header_found = False

            # else:
            #     print("failing to be in waiting")

