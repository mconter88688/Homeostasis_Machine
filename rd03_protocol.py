import serial
from dataclasses import dataclass
from typing import Optional
import constants as cons
import threading


@dataclass
class RadarTarget:
    """Represents a single radar target's data"""
    x_coord: float      # mm, positive or negative
    y_coord: float      # mm, positive or negative
    speed: float        # cm/s, positive or negative
    distance: float     # mm, pixel distance value

class RD03Protocol:
    HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
    FOOTER = bytes([0x55, 0xCC])
    TARGET_DATA_SIZE = 8
    MAX_TARGETS = 3
    
    WAITING_HEADER = 0
    READING_DATA = 1
    WAITING_FOOTER = 2
    
    # Number of positions to keep in trace history
    TRACE_LENGTH = 20

    def __init__(self):
        """Initialize the RD03D Protocol handler with serial port settings"""
        self.serial = None
        self._state = self.WAITING_HEADER
        self._buffer = bytearray()
        self.baudrate = 256000
        self.port = cons.RD03D_PORT
        self.thread = None
        self.running = False
        self.lock = threading.Lock() # avoid race conditions in reading
        self.latest_data = None
   
    def start(self):
        if self.serial is None or not self.serial.is_open:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=cons.TIMEOUT) #wait 1 sec for data
            print("rd03 serial opened!")
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.read_frame, daemon=True)
            self.thread.start()
            print("rd03 thread started!")
    
    def _decode_raw(self, value: int) -> float:
        """Decode a coordinate value according to the protocol specification"""
        # Check if highest bit is set (positive/negative indicator)
        is_negative = not bool(value & 0x8000)
        # Get absolute value (15 bits)
        abs_value = value & 0x7FFF
        return -abs_value if is_negative else abs_value

    def _parse_target_data(self, data: bytes) -> Optional[RadarTarget]:
        """Parse 8 bytes of target data into a RadarTarget object"""
        if all(b == 0 for b in data):  # Check if target data is all zeros
            return None
            
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

    def get_scan(self):
        with self.lock:
            if not self.latest_data: 
                return None 
            else:
                targets = []
                for target in self.latest_data:
                    targets.append(RadarTarget(
                        x_coord=target.x_coord,
                        y_coord=target.y_coord,
                        speed=target.speed,
                        distance=target.distance  
                    ))
                return targets
        


    def read_frame(self):
        """Read and parse a complete data frame from the radar"""
        print("read_frame running")
        frame_data = bytearray()
        header_found = False
        
        while self.running:
            if self.serial.in_waiting:
                byte = ord(self.serial.read())
                print(hex(byte))
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
                    if len(frame_data) == (4 + 24 + 2):  # Header + 3*8 bytes data + Footer
                        #print("complete frame")
                        if frame_data[-2:] == bytes([0x55, 0xCC]):
                            # Valid frame received, parse targets
                            print("valid frame")
                            targets = []
                            data_start = 4  # After header
                            
                            for i in range(3):  # 3 possible targets
                                target_data = frame_data[data_start + i*8:data_start + (i+1)*8]
                                target = self._parse_target_data(target_data)
                                if target is not None:
                                    print("target " + str(i) + " found")
                                    targets.append(target)
                            
                            with self.lock:
                                self.latest_data = targets    
                            
                            
                        else:
                            print("invalid frame")
                            # Invalid frame, start over
                            frame_data = bytearray()
                            header_found = False

            # else:
            #     print("failing to be in waiting")

    def stop(self):
        """Close the serial port"""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.serial = None

