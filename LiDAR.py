import serial
import threading
import struct
import time

TIMEOUT = 1
POINTS = 8
PACKET_LENGTH = 42

# Each point has 3 bytes: 2 for distance and 1 for confidence


class LD19:
    def __init__(self):
        self.baud_rate = 115200
        self.port = "/dev/lidar"
        self.serial = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock() # avoid race conditions in reading
        self.latest_data = [[],[],[]]


    def start(self):
        if self.serial is None or not self.serial.is_open:
            self.serial = serial.Serial(self.port, self.baud_rate, timeout=TIMEOUT) #wait 1 sec for data
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
        while self.running:
            print("reading tthread")
            header = self.serial.read(2) # read first 2 bytes
            if header != b'\x54\x2C':
                print("wrong header")
                continue
            packet = header + self.serial.read(PACKET_LENGTH - 2)
            if len(packet) != PACKET_LENGTH:
                print("not the right amount of packets")
                continue
            
            speed = struct.unpack('<H', packet[2:4])[0] / 64.0 # LiDAR's units are 64 ticks per RPM
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


        

    


