import serial
import os
import cv2
import struct
import numpy as np
import constants as cons
from sensor import Sensor
from time import monotonic_ns
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# LIDAR documentation: https://github.com/LudovaTech/lidar-LD19-tutorial
#Baud Rate: 230400
#Data Length: 8 bits
#Stop Bit: 1
#Parity: None
#Flow Control: None

# LIDAR Specs: https://www.waveshare.com/wiki/DTOF_LIDAR_LD19
# Sweep Rate: 10 Hz
# Measuring Radius: 0.02 - 12 m


POINTS = 12
PACKET_LENGTH = 47
MAX_MEAS_RADIUS = 12000
MIN_MEAS_RADIUS = 20
NUM_BINS = 500
MEDIAN_FILTER_KERNEL_SIZE = 6
EMA_ALPHA = 0.3

def within_radius(distance):
    return (distance >= MIN_MEAS_RADIUS and distance <= MAX_MEAS_RADIUS)

def ema(prev_data, new_unsmoothed_data, alpha = EMA_ALPHA):
    if prev_data is None:
        return new_unsmoothed_data
    return alpha * new_unsmoothed_data + (1 - alpha) * prev_data


class LidarData:
    def __init__(self, speed = None):
        self.angles = []
        self.distances = []
        self.intensities = []
        self.speed = speed
        self.speed_samples = []
        self.start_timestamp = None
        self.end_timestamp = None
        self.mid_timestamp = None
        self.prev_distances = None
        self.prev_intensities = None
        


    def append_all_lists(self, angle, distance, intensity, speed):
        self.angles.append(angle)
        self.distances.append(distance)
        self.intensities.append(intensity)
        self.speed_samples.append(speed)


    def clear_all(self):
        self.angles.clear()
        self.distances.clear()
        self.intensities.clear()
        self.speed_samples.clear()
        self.speed = None
        self.end_timestamp = None
        self.start_timestamp = None
        self.mid_timestamp = None


    def copy(self):
        new_obj = LidarData(self.speed)
        new_obj.angles = self.angles.copy()
        new_obj.distances = self.distances.copy()
        new_obj.intensities = self.intensities.copy()
        new_obj.speed_samples = self.speed_samples.copy()
        new_obj.end_timestamp = self.end_timestamp
        new_obj.mid_timestamp = self.mid_timestamp
        new_obj.start_timestamp = self.start_timestamp
        return new_obj
    
    def calc_mid_timestamp(self):
        if self.start_timestamp is None:
            self.mid_timestamp = None #only if no data
        elif self.end_timestamp is None:
            self.mid_timestamp = self.start_timestamp
        else:
            self.mid_timestamp = self.start_timestamp + (self.end_timestamp - self.start_timestamp)/2

    def bin_lidar_data(self):
        angle_edges = np.linspace(0, 360, NUM_BINS+1)
        angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
        interp_distances = np.interp(angle_centers, self.angles, self.distances, period = 360)
        interp_intensities = np.interp(angle_centers, self.angles, self.intensities, period = 360)
        return angle_centers, interp_distances, interp_intensities

    def circular_median_filter(self, data, kernel_size = MEDIAN_FILTER_KERNEL_SIZE, num_data_points = NUM_BINS):
        half_kernel = kernel_size // 2

        filtered = np.zeros_like(data)

        for i in range(num_data_points):
            neighbor_idxs = [(i + j) % num_data_points for j in range(-half_kernel, half_kernel + 1)]
            filtered[i] = np.median(data[neighbor_idxs])
        return filtered
    
    def lidar_dsp(self):
        self.angles, self.distances, self.intensities = self.bin_lidar_data()
        self.distances = self.circular_median_filter(self.distances)
        self.intensities = self.circular_median_filter(self.intensities)
        self.distances = ema(self.prev_distances, self.distances)
        self.intensities=ema(self.prev_intensities, self.intensities)


    def calc_speed(self):
        if self.speed_samples:
            self.speed = float(np.mean(self.speed_samples))
        else:
            self.speed = None

    def graph(self):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex = True)

        # Raw data
        axs[0].plot(self.angles, self.distances, label="Distance", color="red")
        axs[0].set_title("Distance Data")
        axs[0].set_xlabel("Angle (degrees)")
        axs[0].set_ylabel("Distance (m)")
        axs[0].grid(True)
        axs[0].legend()

        # Filtered data
        axs[1].plot(self.angles, self.intensities, label="Intensities", color="blue")
        axs[1].set_title("Intensity Data")
        axs[1].set_xlabel("Angle (degrees)")
        axs[1].set_ylabel("Intensity")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        graph_path = os.path.join(os.getcwd(), "temp_lidar.png")
        plt.savefig(graph_path)

        img = cv2.imread(graph_path)
        if img is not None:
            cv2.imshow("Graph", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


crc_table = np.array([
    0x00, 0x4d, 0x9a, 0xd7, 0x79, 0x34, 0xe3, 0xae,
    0xf2, 0xbf, 0x68, 0x25, 0x8b, 0xc6, 0x11, 0x5c,
    0xa9, 0xe4, 0x33, 0x7e, 0xd0, 0x9d, 0x4a, 0x07,
    0x5b, 0x16, 0xc1, 0x8c, 0x22, 0x6f, 0xb8, 0xf5,
    0x1f, 0x52, 0x85, 0xc8, 0x66, 0x2b, 0xfc, 0xb1,
    0xed, 0xa0, 0x77, 0x3a, 0x94, 0xd9, 0x0e, 0x43,
    0xb6, 0xfb, 0x2c, 0x61, 0xcf, 0x82, 0x55, 0x18,
    0x44, 0x09, 0xde, 0x93, 0x3d, 0x70, 0xa7, 0xea,
    0x3e, 0x73, 0xa4, 0xe9, 0x47, 0x0a, 0xdd, 0x90,
    0xcc, 0x81, 0x56, 0x1b, 0xb5, 0xf8, 0x2f, 0x62,
    0x97, 0xda, 0x0d, 0x40, 0xee, 0xa3, 0x74, 0x39,
    0x65, 0x28, 0xff, 0xb2, 0x1c, 0x51, 0x86, 0xcb,
    0x21, 0x6c, 0xbb, 0xf6, 0x58, 0x15, 0xc2, 0x8f,
    0xd3, 0x9e, 0x49, 0x04, 0xaa, 0xe7, 0x30, 0x7d,
    0x88, 0xc5, 0x12, 0x5f, 0xf1, 0xbc, 0x6b, 0x26,
    0x7a, 0x37, 0xe0, 0xad, 0x03, 0x4e, 0x99, 0xd4,
    0x7c, 0x31, 0xe6, 0xab, 0x05, 0x48, 0x9f, 0xd2,
    0x8e, 0xc3, 0x14, 0x59, 0xf7, 0xba, 0x6d, 0x20,
    0xd5, 0x98, 0x4f, 0x02, 0xac, 0xe1, 0x36, 0x7b,
    0x27, 0x6a, 0xbd, 0xf0, 0x5e, 0x13, 0xc4, 0x89,
    0x63, 0x2e, 0xf9, 0xb4, 0x1a, 0x57, 0x80, 0xcd,
    0x91, 0xdc, 0x0b, 0x46, 0xe8, 0xa5, 0x72, 0x3f,
    0xca, 0x87, 0x50, 0x1d, 0xb3, 0xfe, 0x29, 0x64,
    0x38, 0x75, 0xa2, 0xef, 0x41, 0x0c, 0xdb, 0x96,
    0x42, 0x0f, 0xd8, 0x95, 0x3b, 0x76, 0xa1, 0xec,
    0xb0, 0xfd, 0x2a, 0x67, 0xc9, 0x84, 0x53, 0x1e,
    0xeb, 0xa6, 0x71, 0x3c, 0x92, 0xdf, 0x08, 0x45,
    0x19, 0x54, 0x83, 0xce, 0x60, 0x2d, 0xfa, 0xb7,
    0x5d, 0x10, 0xc7, 0x8a, 0x24, 0x69, 0xbe, 0xf3,
    0xaf, 0xe2, 0x35, 0x78, 0xd6, 0x9b, 0x4c, 0x01,
    0xf4, 0xb9, 0x6e, 0x23, 0x8d, 0xc0, 0x17, 0x5a,
    0x06, 0x4b, 0x9c, 0xd1, 0x7f, 0x32, 0xe5, 0xa8
], dtype=np.uint8)


def cal_CRC8(packet):
    return_crc = 0
    for byte in packet:
        return_crc = crc_table[(return_crc ^ (byte)) & 0xFF]
    return return_crc

    


class LD19(Sensor):
    def __init__(self):
        super().__init__(name = "LiDAR", baudrate=230400, port=cons.LIDAR_PORT)


    def start(self):
        super().start(bytesize = serial.EIGHTBITS, 
                      parity = serial.PARITY_NONE, 
                      stopbits = serial.STOPBITS_ONE, 
                      xonxoff = False, 
                      rtscts = False, 
                      timeout = cons.TIMEOUT
                      )
        
    
    def send_scan_calc_speed_and_clear(self, return_lidar_data):
        return_lidar_data.calc_mid_timestamp()
        return_lidar_data.calc_speed()
        return_lidar_data.lidar_dsp()
        # print("send_scan:")
        # print(return_lidar_data.angles[0])
        # print(return_lidar_data.angles[-1])
        if return_lidar_data:
            with self.lock:
                self.latest_data = return_lidar_data.copy()
        return_lidar_data.clear_all()
        
        

    
    def _reader_thread(self):
        first_byte = None
        return_lidar_data = LidarData()
        prev_angle = None
        prev_unwrapped = None
        cur_rotation_index = None
        while self.running:
            if not first_byte:
                first_byte = self.serial.read(1)
            if first_byte !=b'\x54':
                first_byte = None
                continue
            second_byte = self.serial.read(1) # read first 2 bytes
            if second_byte != b'\x2C':
                #print(first_byte + second_byte)
                first_byte = second_byte
                continue
            packet = bytearray(first_byte + second_byte)
            packet += self.serial.read(PACKET_LENGTH - 2)
            #print("good header!")
            first_byte = None
            if len(packet) != PACKET_LENGTH:
                print("not the right amount of packets")
                self.serial.reset_input_buffer()
                continue
            #print("good length")
            speed = struct.unpack('<H', packet[2:4])[0] / 360.0  # rotations per second
            start_angle = struct.unpack('<H', packet[4:6])[0] / 100.0 # LiDAR's units are degrees * 100
            end_angle = struct.unpack('<H', packet[PACKET_LENGTH-5:PACKET_LENGTH-3])[0] / 100.0
            #timestamp = struct.unpack('<H', packet[PACKET_LENGTH-3: PACKET_LENGTH-1])[0]
            crc_check = packet[PACKET_LENGTH-1]
            if cal_CRC8(packet[0:PACKET_LENGTH-1]) != crc_check:
                print("Incorrect checksum")
                continue

            angle_diff = (end_angle - start_angle + 360.0) % 360.0
            angle_increment = angle_diff / (POINTS - 1.0) # angle increment between 8 points

            for i in range(POINTS):
                offset = 6 + i * 3
                distance = struct.unpack('<H', packet[offset:offset+2])[0] # unit is mm
                if not within_radius(distance):
                    continue
                intensity = packet[offset+2]
                angle = (start_angle + i * angle_increment) % 360.0
                
                if prev_angle is None:
                    unwrapped = angle
                    cur_rotation_index = math.floor(unwrapped / 360.0)
                else:
                    delta = angle - prev_angle
                    if delta < -180:
                        delta+=360.0
                    elif delta > 180:
                        delta-=360.0
                    unwrapped = prev_unwrapped + delta

                rotation_idx = math.floor(unwrapped/360.0)

                if rotation_idx != cur_rotation_index:
                    return_lidar_data.end_timestamp = monotonic_ns()
                    if return_lidar_data.angles:
                        self.send_scan_calc_speed_and_clear(return_lidar_data)
                    return_lidar_data.start_timestamp = monotonic_ns()
                    cur_rotation_index = rotation_idx
                
                return_lidar_data.append_all_lists(angle, distance, intensity, speed)

                prev_angle = angle
                prev_unwrapped = unwrapped

            
            


        

    


