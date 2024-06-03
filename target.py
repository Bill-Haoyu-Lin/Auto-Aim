import CvCmdApi
import time

class Target:

    def __init__(self):
        self.target_angle = [0, 0, 0]
        self.enemy_detected = False
        self.Global_xyz_filtered = [0, 0, 0]
        self.CvCmder = CvCmdApi.CvCmdHandler('/dev/ttyTHS1')
        self.pitch_lower_limit = -1000
    
    def set_target_angle(self, target_angle):
        if target_angle[1] <  self.pitch_lower_limit:
            target_angle[1] =  self.pitch_lower_limit
        self.target_angle = target_angle
    
    def set_pitch_lower_limit(self, lower_limit):
        self.pitch_lower_limit = lower_limit
        
    def is_detected(self, enemy_detected):
        self.enemy_detected = enemy_detected

    def get_target_angle(self):
        return self.target_angle

    def get_enemy_detected(self):
        return self.enemy_detected

    # Heartbeat from CV to Control
    def call_heartBeat(self):
        while True:
            if self.target_angle[1] <  self.pitch_lower_limit:
                self.target_angle[1] =  self.pitch_lower_limit
            print(self.target_angle[2])
            if self.enemy_detected:
                self.CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=self.target_angle[1], gimbal_yaw_target=self.target_angle[2], chassis_speed_x=0, chassis_speed_y=0)
            else:
                self.target_angle[2] += 0.04
                self.CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=self.target_angle[1], gimbal_yaw_target=self.target_angle[2], chassis_speed_x=0, chassis_speed_y=0)
            time.sleep(1/500)