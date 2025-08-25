import pickle
import os



class Data:
    def __init__(self):
        self.normal_data = []
        self.ld_normal_data = []
        self.rd03_normal_data = []
        self.program_running = True
    
    def load_data(self, feedback_file):
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, "rb") as f:
                    dict_data = pickle.load(f)
                    self.normal_data = dict_data.get("image", [])
                    self.ld_normal_data = dict_data.get("ld", [])
                    self.rd03_normal_data = dict_data.get("rd03", [])
            except (EOFError, pickle.UnpicklingError, AttributeError, ValueError) as e:
                self.normal_data, self.ld_normal_data, self.rd03_normal_data = [], [], []
                print(f"Failed to load data ({type(e).__name__}): {e}")
        else:
            self.normal_data, self.ld_normal_data, self.rd03_normal_data = [], [], []
            print("file does not exist")

    def clear_data(self):
        self.ld_normal_data.clear()
        self.rd03_normal_data.clear()
        self.normal_data.clear()

    def append_normal_data(self, new_data):
        self.normal_data.append(new_data)
    
    def append_ld_normal_data(self, new_data):
        self.ld_normal_data.append(new_data)

    def append_rd03_normal_data(self, new_data):
        self.rd03_normal_data.append(new_data)

    def save_data(self, feedback_file):
        tmp_file = feedback_file + ".tmp"
        try:
            with open(tmp_file, "wb") as f:
                pickle.dump({"image": self.normal_data, 
                            "ld": self.ld_normal_data,
                            "rd03": self.rd03_normal_data}, f)
            os.replace(tmp_file, feedback_file)
        finally:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass

    def is_empty(self):
        return (len(self.normal_data) == 0 and len(self.ld_normal_data) == 0 and len(self.rd03_normal_data) == 0)
    
