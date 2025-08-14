import LiDAR as ld
import rd03_protocol as rd
import camera as cam
from typing import Optional, List
from dataclasses import dataclass
import constants as cons
#from time import sleep



@dataclass
class AllSensorsData:
    camera_data: cam.CameraData
    lidar_data: Optional[ld.LiDARPreprocessedData] = None
    rd03_data: Optional[rd.RadarPreprocessedData] = None
    

    


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
                pass
                # lidar_scan.graph()
                #print(lidar_scan.timestamp)
                # print("Allsensors:")
                # print(lidar_scan.angles[0])
                # print(lidar_scan.angles[-1])
                # for i in range(len(lidar_scan.angles)):
                #     print(str(lidar_scan.angles[i])) # + ", " +  str(lidar_scan.distances[i]) + ", " + str(lidar_scan.intensities[i]))
                # if not lidar_scan.mid_timestamp in self.lidar.timestamp_data:
                print("Num points: " + str(len(lidar_scan.angle_array)))
                print("Timestamp: " + str(lidar_scan.timestamp))
                print(lidar_scan.angle_array[0])
                print(lidar_scan.angle_array[-1])
                #     self.lidar.timestamp_data.append(lidar_scan.mid_timestamp)
                #     if len(lidar_scan.angles) > 506:
                #         for i in range(len(lidar_scan.angles)):
                #             print(str(lidar_scan.angles[i]))
                print("****************")               
            else:
                #print("no lidar readings")
                return None
        if self.rd03:
            try: 
                targets = self.rd03.get_scan()
                if not targets:
                    print("No targets found")
                    return None
                else:
                    for i in range(cons.RADAR_MAX_TARGETS):
                        print(f"Target at ({targets.x_coords[i]}, {targets.y_coords[i]}), Speed: {targets.speeds[i]}")
            except Exception as e:
                print(f"[Radar Error] {e}")
        return AllSensorsData(lidar_data = lidar_scan, rd03_data = targets, camera_data=camera_data)


    
