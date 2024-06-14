from ultralytics import YOLO
import numpy as np
import math
import cv2
from threading import Thread

import CamDataCap
import CvHelper
import target
import time

import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter as EKF 
from filterpy.common import Q_discrete_white_noise

def main():

    debug_mode = False

    model = YOLO("./models/best.engine")

    enemy = target.Target()
    cam = CamDataCap.depth_camera()

    #initialization of imu 
    frame = {"video": None, "disparity": None}

    last = [0,0,0]

    Global_xyz = [0,0,0]

    angle_record = []
    location_record = []
    theta_record = []
    target_record = []

    Global_xyz_filtered = [0,0,0]

    target_yaw_record = np.zeros(1)
    target_pitch_record = np.zeros(1)
    yaw_record = np.zeros(1)

        # Initialize the EKF
    ekf = EKF(dim_x=2, dim_z=1)

    # Initial state
    ekf.x = np.array([0.0, 0.0])  # Starting with angle 0 and small angular velocity
    ekf.F = np.eye(2)             # State transition matrix (will be updated)
    ekf.R = np.array([[0.1]])     # Measurement noise covariance
    ekf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.01)  # Process noise covariance
    ekf.P *= 10                   # Initial state covariance


    thread2 = Thread(target = enemy.call_heartBeat,args=())
    thread2.start()
    print("heartbeat start")

    thread1 = Thread(target = cam.start,args=())
    thread1.start()
    print("Camera Thread start")
    detected = False
    time.sleep(4)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
    out = cv2.VideoWriter('output2.mp4', fourcc, 30.0, (640, 360))


    enemy.set_pitch_lower_limit(cam.angles_min[1])
    enemy.ack_start(True)
   
    cur_time = time.time()
    while True:
        frame = cam.frame
        out.write(frame["video"])
        
        #get current imu angle
        dt = time.time()-cur_time

        angle_rt= cam.cur_angle
        cur_angle = - np.array(angle_rt) + np.array(cam.angles_default)
        cur_angle[2] = CvHelper.wrap_angle(cur_angle[2])
        yaw_record = np.append(yaw_record,cur_angle[2])

        detected = False
        if not frame["video"] is None:

            if frame["video"].shape == (360,640,3):

                if debug_mode:
                    start_time = time.time()  # Record the start time
                    results = model.track(frame["video"],verbose=False,device="cuda",batch=16,persist=True,task='detect',conf=0.6)[0]
                    end_time = time.time() 
                    elapsed_time = end_time - start_time 
                    print('video time: ', elapsed_time)
                else:
                    results = model.track(frame["video"],verbose=False,device="cuda",batch=16,persist=True,task="detect",conf=0.6)[0]

                for result in results:
                    if result != None:
                       
                        angle_record.append(cur_angle[2])

                        boxes_xy = result.boxes.xyxy[0]

                        center_x = int((boxes_xy[0]+boxes_xy[2])/2)
                        center_y = int((boxes_xy[1]+boxes_xy[3])/2)
                        
                        if enemy.is_enemy(frame["video"][int(boxes_xy[0]):int(boxes_xy[2]),int(boxes_xy[1]):int(boxes_xy[3])]):
                            detected = True  
                        else:
                            detected = False
                            
                        enemy.is_detected(detected)
                        # print(center_x, center_y)
                        # topLeft = dai.Point2f(((4*(center_x-7))/4056), ((((center_y-7)+30)*4+220)/3040))# (960,540))[ 30:510, 160:800]
                        # bottomRight = dai.Point2f(((4*(center_x+7))/4056),((((center_y+7)+30)*4+220)/3040))

                        frame["video"] = cv2.circle(frame["video"], (center_x,center_y), 10, (255, 0, 0) , 2)
                        
                        focal_length_in_pixel = 1980*0.65* 4.81 / 11.043412
                        focal_length_in_pixel_pitch =  2.3*1080 * 4.81 / 11.043412
                        
                        theta  = math.atan((center_x - 320)/focal_length_in_pixel)
                        phi = -math.atan((center_y - 180)/focal_length_in_pixel_pitch)
                        print(theta)

                        # print("angle 2",cur_angle[2])
                        # print("angle 1",cur_angle[1])

                    
                        #roll pitch yaw
                        # Yaw clockwise +
                        # Pitch Up +
                        #print("theta is: =",theta)
                        Global_xyz[2]= cur_angle[2]+theta
                        Global_xyz[1]= cur_angle[1]+phi

                        target_yaw_record = np.append(target_yaw_record,Global_xyz[2])
                        yaw_record = np.append(yaw_record,Global_xyz[2])

                        target_pitch_record = np.append(target_pitch_record,Global_xyz[1])

                        # print(target_yaw_record[-1])
                        
                        theta_record.append(theta)
                        location_record.append(Global_xyz[2])
                        target_record.append(Global_xyz_filtered[2])

                        ekf.F = CvHelper.jfx(ekf.x, dt)
                        ekf.predict()
                        ekf.update(np.array(yaw_record[-1]), HJacobian=CvHelper.jhx, Hx=CvHelper.hx)
                        ekf.x[0] = CvHelper.normalize_angle(ekf.x[0])  # Normalize the angle

                        # Global_xyz_filtered[2] = CvHelper.wrap_angle(Global_xyz[2])
                        Global_xyz_filtered[2] = ekf.x[0]
                        Global_xyz_filtered[1] = target_pitch_record[-1]+0.1
                        
                        Global_xyz_filtered = np.float32(Global_xyz_filtered)
                        enemy.set_target_angle(Global_xyz_filtered)

                        #print global location for debug

                        
                        last = Global_xyz
                        
                        break
                    else:
                        enemy.is_detected(False)
                
                if not detected:
                    enemy.is_detected(False)

        cv2.imshow("video", frame["video"])

        if cv2.waitKey(1) == ord('q'):
            break

        cur_time = time.time()

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
