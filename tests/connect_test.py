import CvCmdApi

# Example usage
CvCmder = CvCmdApi.CvCmdHandler('/dev/ttyTHS0')
while True:
        CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=0, gimbal_yaw_target=1, chassis_speed_x=0, chassis_speed_y=0)
