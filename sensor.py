import threading
import serial

class Sensor:
    def __init__(self, name, baudrate, port):
        self.name = name
        self.serial = None
        self.baudrate = baudrate
        self.port = port
        self.thread = None
        self.running = False
        self.lock = threading.Lock() # avoid race conditions in reading
        self.latest_data = None

    def start(self, bytesize, parity, stopbits, xonxoff = False, rtscts = False, timeout = 1):
        if self.serial is None or not self.serial.is_open:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                xonxoff=xonxoff,
                rtscts=rtscts,
                timeout=timeout) 
            print(self.name + " serial opened!")
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._reader_thread, daemon=True)
            self.thread.start()
            print(self.name + " thread started!")

    