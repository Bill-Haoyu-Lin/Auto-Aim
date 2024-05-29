from ultralytics import YOLO
import numpy as np
from numpy.linalg import inv

from datetime import timedelta
import math
import cv2
import time
from threading import Thread

import CamDataCap
import CvHelper
import target

import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter as EKF 
from filterpy.common import Q_discrete_white_noise

def main():
    model = YOLO("./models/best.pt")
    targets = target.Target()
    cam = CamDataCap.depth_camera()

    #initialization of imu 
    angles_default = [0,0,0]
    init_state = True
    frame = {"video": None, "disparity": None}
    angles = [0,0,0]
    frame_type  = 'disparity'
    last = [0,0,0]
    enemy_detected = False

    coordinates_3d = [0,0,0]
    Global_xyz = [0,0,0]

    angle_record = []
    location_record = []
    frame_xyz_record = []
    theta_record = []
    target_record = []

    angle_reached = True
    target_angle = [0,0,0]
    Global_xyz_filtered = [0,0,0]

    target_yaw_record = np.empty(60)
    target_pitch_record = np.empty(1)

    thread2 = Thread(target = targets.call_heartBeat,args=())
    thread2.start()
    print("heartbeat start")

    thread1 = Thread(target = cam.start,args=())
    thread1.start()
    print("Camera Thread start")

    while True:

        if not frame["video"] is None:

            if frame["video"].shape == (480,640,3):
                # print("frame recived")
                
                results = model(frame["video"],verbose=False)[0]
                for result in results:
                    if result != None:

                        enemy_detected = True
                        boxes_xy = result.boxes.xyxy[0]

                        center_x = int((boxes_xy[0]+boxes_xy[2])/2)
                        center_y = int((boxes_xy[1]+boxes_xy[3])/2)
                        
                        # print(center_x, center_y)
                        # topLeft = dai.Point2f(((4*(center_x-7))/4056), ((((center_y-7)+30)*4+220)/3040))# (960,540))[ 30:510, 160:800]
                        # bottomRight = dai.Point2f(((4*(center_x+7))/4056),((((center_y+7)+30)*4+220)/3040))
                        

                        frame["video"] = cv2.circle(frame["video"], (center_x,center_y), 10, (255, 0, 0) , 2)
                        frame["disparity"] = cv2.circle(frame["disparity"], (int((((4*(center_x+160))/4056)*640)),int(400*(((center_y+30)*4+220)/3040))), 60, (255, 255, 255) , 2)
                        
                        focal_length_in_pixel = 1280 * 4.81 / 11.043412
                        focal_length_in_pixel_pitch =  1.8*960 * 4.81 / 11.043412
                        
                        theta  = math.atan((center_x - 320)/focal_length_in_pixel)
                        phi = math.atan((center_y - 240)/focal_length_in_pixel_pitch)
                        
                        theta_record.append(theta)

                        # if abs(theta) >0.25:
                        #     theta = theta * 2

                        angle_rt= angles.copy()
                        cur_angle = np.array(angle_rt) - np.array(angles_default)

                        cur_angle[2] = CvHelper.wrap_angle(-cur_angle[2])
                        angle_record.append(cur_angle[2])

                        #roll pitch yaw
                        # Yaw clockwise + 
                        # Pitch down +
                        #print("theta is: =",theta)
                        Global_xyz[2]= cur_angle[2]+theta
                        Global_xyz[1]= phi - cur_angle[1]

                        target_yaw_record = np.append(target_yaw_record,Global_xyz[2])
                        target_pitch_record = np.append(target_pitch_record,Global_xyz[1])


                        #Numbers for tunning 
                        b,a= signal.ellip(3, 0.04, 60, 0.125)
                        target_yaw_record = signal.filtfilt(b, a,target_yaw_record,method="gust", irlen=70)
                        target_pitch_record = signal.filtfilt(b, a,target_pitch_record,method="gust")
                        
                        
                        target_yaw_window = target_yaw_record[-10:]
                        z = np.polyfit([0,1,2,3,4,5,6,7,8,9],target_yaw_window,3)
                        pred = z[0]*10 + z[1]
                        
                        if z[0] >5:
                            target_yaw_record[-1] *= 1.5

                        Global_xyz_filtered[2] = target_yaw_record[-1]-0.15
                        # Global_xyz_filtered[2] = pred -0.1
                        Global_xyz_filtered[1] = target_pitch_record[-1]-0.1
                    
                        #Numbers for tunning 
                        b,a= signal.ellip(3, 0.04, 60, 0.125)
                        target_yaw_record = signal.filtfilt(b, a,target_yaw_record,method="gust", irlen=70)
                        target_pitch_record = signal.filtfilt(b, a,target_pitch_record,method="gust")
                        # Initialize the EKF

                        ekf = EKF(dim_x=1, dim_z=1)

                        # Initial state
                        ekf.x = np.array([0.0])  # Starting angle
                        ekf.F = np.array([[1]])  # State transition matrix
                        ekf.R = np.array([[0.1]])  # Measurement noise covariance
                        ekf.Q = Q_discrete_white_noise(dim=1, dt=1, var=0.05)  # Process noise covariance
                        ekf.P *= 1000  # Initial state covariance

                        # Example measurements (angles from -pi to pi)
                        measurements = target_yaw_record[-10:]   # Example angle measurements in radians
                        dt = 1.0  # Time step

            
                        Global_xyz_filtered[2] = target_yaw_record[-1]-0.15
                        Global_xyz_filtered[1] = target_pitch_record[-1]-0.1

                        #print global location for debug
                        location_record.append(Global_xyz[2])
                        target_record.append(Global_xyz_filtered[2])
                        
                        if angle_reached : 
                            target_angle = Global_xyz 

                        #print(f"Global X: {Global_xyz[0]} mm, Y: {Global_xyz[1]} mm, Z: {Global_xyz[2]} mm")
                        last = Global_xyz
                        
                        break
                    else:
                        Global_xyz_filtered[2]+= 0.1
                        enemy_detected= False
        cv2.imshow("video", frame["video"])
        cv2.imshow("depth", frame["disparity"])

        if cv2.waitKey(1) == ord('q'):
            break


    length = len(angle_record)
    t = [(i + 1) / 100 for i in range(length)]
    fig, ax = plt.subplots()

    ax.plot(t, angle_record,label='IMU angle')
    ax.plot(t,location_record,label='global angle')
    ax.plot(t,theta_record,label='delta angle')
    ax.plot(t,target_record,label='target angle')

    ax.set(xlabel='time (s)', ylabel='angle',
        title='angle(rad) vs time')
    ax.legend()
    ax.grid()

    fig.savefig("test.png")
    plt.show()
    
    #kill the camera thread    
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()