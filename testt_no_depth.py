import depthai as dai
from ultralytics import YOLO
import numpy as np
from numpy.linalg import inv

from datetime import timedelta
import math
import cv2
import time
from threading import Thread
import CvCmdApi

import matplotlib.pyplot as plt
from scipy import signal

def spherical_to_cartesian(r, phi, theta):
    phi = ((phi + np.pi) % (2*np.pi)) - np.pi

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, -y, z

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        #OUTPUT --> i : yaw, j : roll, k : pitch 
        return [roll_x, pitch_y, yaw_z] # in radians

def wrap_angle(angle):
    # Wrap angle to range -pi to pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

def tramform2base(x,y,z,angles):
    x = x+20 # 200 mm offset from camera to type-c board
    pitch = round(angles[1], 3)
    roll = round(angles[0], 3)
    yaw = -round(angles[2], 3)
    #transform z --> x, x --> y, y --> -z to standard TF
    R_pitch = np.array([[math.cos(pitch ), 0, math.sin(pitch )],
                    [0, 1, 0],
                    [-math.sin(pitch ), 0, math.cos(pitch )]])
    
    R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])
    
    xyz = np.array([x,y,z])
    xyz_final = (R_yaw.dot(R_pitch.dot(xyz)))
    # print(xyz_final)
    x_final = xyz_final[0]
    y_final = xyz_final[1]
    z_final = xyz_final[2]
    
    #calculate pitch yaw roll final
    yaw_final = np.arctan2(y_final, x_final)
    # if yaw_final <= 0:
    #     yaw_final += math.pi
    # else:
    #     yaw_final -= math.pi
    pitch_final = np.arctan2(z_final, np.sqrt(x_final**2 + y_final**2))
    roll_final = np.arctan2(np.sin(yaw)*z_final - np.cos(yaw)*y_final, np.cos(pitch)*x_final + np.sin(pitch)*y_final)
    # print([roll_final, pitch_final, yaw_final])
    # Convert angles to degree
    return [round(roll_final, 2), 
            round(pitch_final, 2), 
            round(yaw_final, 2)]

def get_frame_from_camera():    
    with dai.Device(pipeline) as device:
        global frame
        global angles
        global angles_default
        global init_state
        global frame_type 
        global topLeft, bottomRight
        global coordinates_3d
        global cfg

        queue = device.getOutputQueue("xout", 10, False)

        depthQueue = device.getOutputQueue(name="depth", maxSize=2, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=2, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
        imuQueue = device.getOutputQueue(name="imu", maxSize=2, blocking=False)
        baseTs = None
        while True:
            #get RGB data
            msgGrp = queue.get()

            #get IMU data
            imuData = imuQueue.get()
            imuPackets = imuData.packets

            #Get depth data
            inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
            depthFrame = inDepth.getFrame() # depthFrame values are in millimeters
            depth_downscaled = depthFrame[::4]

            if np.all(depth_downscaled == 0):
                min_depth = 0  # Set a default minimum depth value when all elements are zero
            else:
                min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
            max_depth = np.percentile(depth_downscaled, 99)
            depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            spatialData = spatialCalcQueue.get().getSpatialLocations()
            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                
                #transform z --> x, x --> y, y --> -z to standard TF
                coordinates_3d = [int(depthData.spatialCoordinates.z),int(depthData.spatialCoordinates.x) ,-int(depthData.spatialCoordinates.y)]
                # print(f"X: {int(depthData.spatialCoordinates.x)} mm, Y: {int(depthData.spatialCoordinates.y)} mm, Z: {int(depthData.spatialCoordinates.z)} mm")
            
            #unpack RGB data
            for name, msg in msgGrp:
                frame_get = msg.getCvFrame()
                frame_type = name
                # print(f"Received {name} frame")
                if name == "disparity":
                    frame["disparity"] = (frame_get * disparityMultiplier).astype(np.uint8)
                    frame["disparity"]= cv2.applyColorMap(frame["disparity"], cv2.COLORMAP_JET)
                else:
                    frame["video"] = cv2.resize(frame_get, (960,540))[ 30:510, 160:800]


        
            #unpack IMU data
            for imuPacket in imuPackets:
                rVvalues = imuPacket.rotationVector

                rvTs = rVvalues.getTimestampDevice()
                if baseTs is None:
                    baseTs = rvTs
                rvTs = rvTs - baseTs

                imuF = "{:.06f}"
                tsF  = "{:.03f}"
                #i : yaw, j : roll, k : pitch
                angles = euler_from_quaternion(rVvalues.i,rVvalues.j,rVvalues.k,rVvalues.real)
                
                if init_state:
                    angles_default = angles
                    init_state = False
            
                # print(f"i: {tsF.format(angles[0])} j: {tsF.format(angles[1])} "
                #     f"k: {tsF.format(angles[2])}",end='\r')
                
            config.roi = dai.Rect(topLeft, bottomRight)
            config.calculationAlgorithm = calculationAlgorithm
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)

def call_heartBeat():
    global enemy_detected
    global Global_xyz
    global last
    global target_angle
    global Global_xyz_filtered
    while True:
        print(target_angle[1])
        if enemy_detected:
            CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=Global_xyz_filtered[1], gimbal_yaw_target=Global_xyz_filtered[2], chassis_speed_x=0, chassis_speed_y=0)
        else:
            CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=Global_xyz_filtered[1], gimbal_yaw_target=Global_xyz_filtered[2], chassis_speed_x=0, chassis_speed_y=0)
        time.sleep(1/500)

model = YOLO("best.pt")
CvCmder = CvCmdApi.CvCmdHandler('/dev/ttyUSB0')

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
color = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
imu = pipeline.create(dai.node.IMU)
sync = pipeline.create(dai.node.Sync)

color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color.setFps(60)

xoutGrp = pipeline.create(dai.node.XLinkOut)

#Add for depth video sync
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

#Add for IMU Xlink
xlinkOut = pipeline.create(dai.node.XLinkOut)

xoutGrp.setStreamName("xout")

#Add for depth video sync
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xlinkOut.setStreamName("imu")

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)

#config roi
topLeft = dai.Point2f(0.45, 0.45)
bottomRight = dai.Point2f(0.55, 0.55)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
config.roi = dai.Rect(topLeft, bottomRight)

imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 60)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
imu.out.link(xlinkOut.input)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

color.setCamera("color")

sync.setSyncThreshold(timedelta(milliseconds=10))

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


stereo.disparity.link(sync.inputs["disparity"])
color.video.link(sync.inputs["video"])

sync.out.link(xoutGrp.input)

disparityMultiplier = 255.0 / stereo.initialConfig.getMaxDisparity()


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

thread2 = Thread(target = call_heartBeat,args=())
thread2.start()
print("heartbeat start")

thread1 = Thread(target = get_frame_from_camera,args=())
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
                    topLeft = dai.Point2f(((4*(center_x-7))/4056), ((((center_y-7)+30)*4+220)/3040))# (960,540))[ 30:510, 160:800]
                    bottomRight = dai.Point2f(((4*(center_x+7))/4056),((((center_y+7)+30)*4+220)/3040))
   
 
                    frame["video"] = cv2.circle(frame["video"], (center_x,center_y), 10, (255, 0, 0) , 2)
                    frame["disparity"] = cv2.circle(frame["disparity"], (int((((4*(center_x+160))/4056)*640)),int(400*(((center_y+30)*4+220)/3040))), 60, (255, 255, 255) , 2)
                    
                    focal_length_in_pixel = 1280 *0.95* 4.81 / 11.043412
                    focal_length_in_pixel_pitch =  1.8*960 * 4.81 / 11.043412
                    
                    theta  = math.atan((center_x - 320)/focal_length_in_pixel)
                    phi = math.atan((center_y - 240)/focal_length_in_pixel_pitch)
                    
                    theta_record.append(-phi)

                    # if abs(theta)<0.1:
                    #     theta =0
                    angle_rt= angles.copy()
                    cur_angle = np.array(angle_rt) - np.array(angles_default)


                    cur_angle[2] = wrap_angle(-cur_angle[2])
                    angle_record.append(cur_angle[1])

                    #roll pitch yaw
                    # Yaw clockwise + 
                    # Pitch down +
                    #print("this is",theta)
                    
                    Global_xyz[2]= cur_angle[2]+theta
                    Global_xyz[1]= phi - cur_angle[1]
                    # Global_xyz = (np.array(Global_xyz)+np.array(last))/2
                    target_yaw_record = np.append(target_yaw_record,Global_xyz[2])
                    target_pitch_record = np.append(target_pitch_record,Global_xyz[1])

                    #Numbers for tunning 
                    b,a= signal.ellip(3, 0.05, 60, 0.125)
                    target_yaw_record = signal.filtfilt(b, a,target_yaw_record,method="gust")
                    target_pitch_record = signal.filtfilt(b, a,target_pitch_record,method="gust")

        
                    Global_xyz_filtered[2] = target_yaw_record[-1]
                    Global_xyz_filtered[1] = target_pitch_record[-1]-0.08

                    #print global location for debug
                    location_record.append(Global_xyz[1])
                    target_record.append(Global_xyz_filtered[1])
                    
                    if angle_reached : 
                        target_angle = Global_xyz 

                    # if (-0.02<theta <0.02) or ((target_angle[2]-0.02)< cur_angle[2] <= (target_angle[2]+0.02)):
                    #     angle_reached = True
                    # else:
                    #     angle_reached = False

                    #print(f"Global X: {Global_xyz[0]} mm, Y: {Global_xyz[1]} mm, Z: {Global_xyz[2]} mm")
                    last = Global_xyz
                    break
                else:
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