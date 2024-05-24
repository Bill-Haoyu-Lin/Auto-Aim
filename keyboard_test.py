import CvCmdApi

# Example usage
CvCmder = CvCmdApi.CvCmdHandler('/dev/ttyUSB0')
# oldflags = (False, False, False)
# while True:
#     flags = CvCmder.CvCmd_Heartbeat(0, 0, 0, 0)  # gimbal_coordinate_x, gimbal_coordinate_y, chassis_speed_x, chassis_speed_y
#     if flags != oldflags:
#         oldflags = flags
#         print(flags)

import keyboard
import time
import threading
oldflags = (False, False, False)
counter = 0
loop_delay = 0.05
chassis_vy = 0
chassis_vx = 0
gimbal_pitch = 0
gimbal_yaw = 0



def send_heartbeat():
    global gimbal_pitch, gimbal_yaw, chassis_vx, chassis_vy
    while True:
        CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=gimbal_pitch, gimbal_yaw_target=gimbal_yaw, chassis_speed_x=chassis_vx, chassis_speed_y=chassis_vy)

thread1 = threading.Thread(target=send_heartbeat, args=())
thread1.start()
print("Press 'q' to quit")


while True:


    input_data = input("Enter a command: ")
    if input_data == "s":
        chassis_vy = -0.3
    elif input_data == "w":
        chassis_vy = 0.3
    elif input_data == "a":
        chassis_vx = -0.3
    elif input_data == "d":
        chassis_vx = 0.3

    # percentage
    if input_data == "k":
        gimbal_pitch = -0.2
    elif  input_data == "i":
        gimbal_pitch = 0.2
    elif  input_data == "j":
        gimbal_yaw = -0.2
    elif  input_data == "l":
        gimbal_yaw = 0.4

    if  input_data == "q":
        break
    
    # if counter == int(5/loop_delay):
    #     CvCmder.CvCmd_Shoot()

    # print(chassis_vx, chassis_vy)
    time.sleep(loop_delay)
    counter += 1
thread1.join()
print("Thread finished...exiting lmaoooo XD very cool")
