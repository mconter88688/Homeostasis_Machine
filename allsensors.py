import LiDAR as ld
import rd03_protocol as rd
import camera as cam
from typing import Optional, List
from dataclasses import dataclass
from time import sleep



@dataclass
class AllSensorsData:
    camera_data: cam.CameraData
    lidar_data: Optional[ld.LidarData] = None
    rd03_data: Optional[List[rd.RadarTarget]] = None
    

    


class AllSensors:
    def __init__(self, lidar_present = True, rd03_present = True, gemini_present = True):
        self.lidar = ld.LD19()  if lidar_present else None
        self.rd03 = rd.RD03Protocol() if rd03_present else None
        self.gemini = cam.Camera() if gemini_present else None
        self.lidar_data = None
        self.rd03_data = None
        self.gemini_data = None

    def start(self):
        if self.gemini:
            self.gemini.configure_streams()
            self.gemini.configure_HDR()
            self.gemini.start()
        if self.lidar:
            self.lidar.start()
        if self.rd03:
            self.rd03.start()

    def stop(self):
        if self.gemini:
             self.gemini.stop()
        if self.lidar:
            self.lidar.stop()
        if self.rd03:
            self.rd03.stop()

    def capture_sensor_info(self):
        camera_data = False
        lidar_scan = False
        targets = False
        if self.gemini:
            camera_data = self.gemini.one_capture()
            if not camera_data:
                return None
        if self.lidar:
            lidar_scan = self.lidar.get_scan()
            if lidar_scan:
                #print(lidar_scan.timestamp)
                # print("Allsensors:")
                # print(lidar_scan.angles[0])
                # print(lidar_scan.angles[-1])
                print("Num points: " + str(len(lidar_scan.angles)))
                print("Timestamp: " + str(lidar_scan.mid_timestamp))
                # for i in range(len(lidar_scan.angles)):
                #     print(str(lidar_scan.angles[i])) # + ", " +  str(lidar_scan.distances[i]) + ", " + str(lidar_scan.intensities[i]))
                print("****************")
            else:
                return None
        if self.rd03:
            try: 
                targets = self.rd03.get_scan()
                if not targets:
                    print("No targets found")
                    return None
                else:
                    for target in targets:
                        print(f"Target at ({target.x_coord}, {target.y_coord}), Speed: {target.speed}")
            except Exception as e:
                print(f"[Radar Error] {e}")
        return AllSensorsData(lidar_data = lidar_scan, rd03_data = targets, camera_data=camera_data)


    
