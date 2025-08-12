import serial
import threading
import struct
import numpy as np
import constants as cons
from sensor import Sensor

# LIDAR documentation: https://github.com/LudovaTech/lidar-LD19-tutorial
#Baud Rate: 230400
#Data Length: 8 bits
#Stop Bit: 1
#Parity: None
#Flow Control: None


POINTS = 12
PACKET_LENGTH = 47

class LidarData:
    def __init__(self, timestamp = None, speed = None):
        self.angles = []
        self.distances = []
        self.intensities = []
        self.speed = []
        self.timestamp = timestamp

    def append_all_lists(self, angle, distance, intensity, speed):
        self.angles.append(angle)
        self.distances.append(distance)
        self.intensities.append(intensity)
        self.speed.append(speed)

    def clear_all(self):
        self.angles.clear()
        self.distances.clear()
        self.intensities.clear()
        self.speed = []
        self.timestamp = None


    def copy(self):
        new_obj = LidarData(self.timestamp, self.speed)
        new_obj.angles = self.angles.copy()
        new_obj.distances = self.distances.copy()
        new_obj.intensities = self.intensities.copy()
        return new_obj







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
        return_lidar_data.speed = np.mean(return_lidar_data.speed)
        with self.lock:
             self.latest_data = return_lidar_data
        return_lidar_data.clear_all()
        
        

    
    def _reader_thread(self):
        first_byte = None
        return_lidar_data = LidarData()
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
            timestamp = struct.unpack('<H', packet[PACKET_LENGTH-3: PACKET_LENGTH-1])[0]
            crc_check = packet[PACKET_LENGTH-1]
            if cal_CRC8(packet[0:PACKET_LENGTH-1]) != crc_check:
                print("Incorrect checksum")
                continue

            angle_diff = (end_angle - start_angle + 360.0) % 360.0
            angle_increment = angle_diff / (POINTS - 1.0) # angle increment between 8 points

            return_lidar_data.timestamp = timestamp
            last_angle = start_angle
            for i in range(POINTS):
                offset = 6 + i * 3
                distance = struct.unpack('<H', packet[offset:offset+2])[0]
                intensity = packet[offset+2]
                angle = (start_angle + i * angle_increment) % 360.0
                if abs(angle - last_angle) > 350:
                    # print("Last angle: " + str(last_angle))
                    # print("Cur angle: " + str(angle))
                    # print("diff: " + str(abs(angle-last_angle)))
                    self.send_scan_calc_speed_and_clear(return_lidar_data)
                return_lidar_data.append_all_lists(angle, distance, intensity, speed)
                last_angle = angle
            
            


        

    


