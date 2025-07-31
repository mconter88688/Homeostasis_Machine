import serial
import threading
import struct
import time

# LIDAR documentation: https://github.com/LudovaTech/lidar-LD19-tutorial
#Baud Rate: 230400
#Data Length: 8 bits
#Stop Bit: 1
#Parity: None
#Flow Control: None

TIMEOUT = 1
POINTS = 12
PACKET_LENGTH = 47
BYTESIZE = serial.EIGHTBITS
STOPBITS = serial.STOPBITS_ONE
PARITY = serial.PARITY_NONE

# Each point has 3 bytes: 2 for distance and 1 for confidence


class LD19:
    def __init__(self):
        self.baud_rate = 230400
        self.port = "/dev/lidar"
        self.serial = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock() # avoid race conditions in reading
        self.latest_data = [[],[],[]]


    def start(self):
        if self.serial is None or not self.serial.is_open:
            self.serial = serial.Serial(port=self.port, 
                                        baudrate=self.baud_rate, 
                                        bytesize=BYTESIZE,
                                        stopbits=STOPBITS,
                                        parity=PARITY,
                                        xonxoff=False,
                                        rtscts=False,
                                        timeout=TIMEOUT) #wait 1 sec for data
            print("serial opened!")
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._reader_thread, daemon=True)
            self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.serial = None
    
    def get_scan(self):
        with self.lock:
            return list(self.latest_data) # make copy to ensure thread safety
        
    def _reader_thread(self):
        first_byte = None
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
            packet = first_byte + second_byte + self.serial.read(PACKET_LENGTH - 2)
            #print("good header!")
            first_byte = None
            if len(packet) != PACKET_LENGTH:
                print("not the right amount of packets")
                continue
            #print("good length")
            speed = struct.unpack('<H', packet[2:4])[0] / 360.0  # rotations per second
            start_angle = struct.unpack('<H', packet[4:6])[0] / 100.0 # LiDAR's units are degrees * 100
            end_angle = struct.unpack('<H', packet[30:32])[0] / 100.0

            angle_diff = (end_angle - start_angle + 360.0) % 360.0
            angle_increment = angle_diff / (POINTS - 1.0) # angle increment between 8 points

            angles = []
            distances = []
            confidences = []
            for i in range(POINTS):
                offset = 6 + i * 3
                distance = struct.unpack('<H', packet[offset:offset+2])[0]
                confidence = packet[offset+2]
                angle = (start_angle + i * angle_increment) % 360.0
                angles.append(angle)
                distances.append(distance)
                confidences.append(confidence)
            
            with self.lock:
                self.latest_data = [angles, distances, confidences]


        

    


