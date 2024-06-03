import CvCmdApi
import time

class Target:

    def __init__(self):
        self.target_angle = [0, 0, 0]
        self.enemy_detected = False
        self.Global_xyz_filtered = [0, 0, 0]
        self.CvCmder = CvCmdApi.CvCmdHandler('/dev/ttyTHS1')
    
    def set_target_angle(self, target_angle):
        self.target_angle = target_angle
        
    def is_detected(self, enemy_detected):
        self.enemy_detected = enemy_detected

    def get_target_angle(self):
        return self.target_angle

    def get_enemy_detected(self):
        return self.enemy_detected

    # Heartbeat from CV to Control
    def call_heartBeat(self):
        while True:
            # print(self.target_angle[1])
            if self.enemy_detected:
                self.CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=self.target_angle[1], gimbal_yaw_target=self.target_angle[2], chassis_speed_x=0, chassis_speed_y=0)
            else:
                self.CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=self.target_angle[1], gimbal_yaw_target=self.target_angle[2], chassis_speed_x=0, chassis_speed_y=0)
            time.sleep(1/500)