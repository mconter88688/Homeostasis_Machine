import threading

class Sensor:
    def __init__(self, baudrate, port):
        self.serial = None
        self.baudrate = baudrate
        self.port = port
        self.thread = None
        self.running = False
        self.lock = threading.Lock() # avoid race conditions in reading
        self.latest_data = None

    