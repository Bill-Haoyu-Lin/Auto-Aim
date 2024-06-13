import CvCmdApi
import time
import cv2
import numpy as np

class Target:

    def __init__(self):
        self.target_angle = [0, 0, 0]
        self.enemy_detected = False
        self.Global_xyz_filtered = [0, 0, 0]
        self.CvCmder = CvCmdApi.CvCmdHandler('/dev/ttyTHS1')
        self.color = self.CvCmder.CvCmd_GetTeamColor()
    
    def set_target_angle(self, target_angle):
        self.target_angle = target_angle
        
    def is_detected(self, enemy_detected):
        self.enemy_detected = enemy_detected

    def get_target_angle(self):
        return self.target_angle

    def get_enemy_detected(self):
        return self.enemy_detected

    def is_enemy(self,crop):
        crop=cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Define color ranges for red and blue
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])

        lower_blue = np.array([100, 120, 70])
        upper_blue = np.array([130, 255, 255])

        # Create masks for each color range
        red_mask = cv2.inRange(crop, lower_red, upper_red)
        blue_mask = cv2.inRange(crop, lower_blue, upper_blue)

        # Count the number of pixels for each color
        red_pixels = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)

        # Determine the dominant color
        if red_pixels > blue_pixels:
            Target = 1   # Red
        elif blue_pixels > red_pixels:
            Target = 0   # Blue

        if Target != self.color:
            return True
    

    # Heartbeat from CV to Control
    def call_heartBeat(self):
        while True:
            # print(self.target_angle[1])
            if self.enemy_detected:
                self.CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=self.target_angle[1], gimbal_yaw_target=self.target_angle[2], chassis_speed_x=0, chassis_speed_y=0)
            else:
                self.CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=self.target_angle[1], gimbal_yaw_target=self.target_angle[2], chassis_speed_x=0, chassis_speed_y=0)
            time.sleep(1/500)