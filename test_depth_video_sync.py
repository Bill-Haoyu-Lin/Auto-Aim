import depthai as dai
from ultralytics import YOLO
import numpy as np


from datetime import timedelta
import math
import cv2
import CvCmdApi
from threading import Thread

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
        return roll_x, pitch_y, yaw_z # in radians

def tramform2base(x,y,z,angles):
    x = x-200 # 200 mm offset from camera to type-c board
    pitch = angles[2]
    roll = angles[1]
    yaw = angles[0]

    #transform z --> x, x --> y, y --> -z to standard TF
    R_pitch = np.array([[math.cos(pitch ), 0, math.sin(pitch ),0],
                    [0, 1, 0,0],
                    [-math.sin(pitch ), 0, math.cos(pitch ),0],
                    [0,0,0,1]])
    
    R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0,0],
                    [math.sin(yaw), math.cos(yaw), 0,0],
                    [0, 0, 1,0],
                    [0,0,0,1]])
    
    xyz = np.array([x,y,z,1]).transpose()
    xyz_final = np.dot(R_yaw,np.dot(R_pitch,xyz)).transpose()
    
    #calculate pitch yaw roll final
    yaw_final = np.arctan2(y, x)
    pitch_final = np.arctan2(-z, np.sqrt(x**2 + y**2))
    roll_final = np.arctan2(np.sin(yaw)*z - np.cos(yaw)*y, np.cos(pitch)*x + np.sin(pitch)*y)
    
    # Convert angles to degrees

    return [roll_final, pitch_final, yaw_final]

def get_frame_from_camera():    
    with dai.Device(pipeline) as device:
        global frame
        global angles
        global angles_default
        global init_state
        global frame_type 
        global topLeft, bottomRight
        global coordinates_3d

        queue = device.getOutputQueue("xout", 10, False)

        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
        imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
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

model = YOLO("best.onnx")
CvCmder = CvCmdApi.CvCmdHandler('/dev/ttyTHS0')

pipeline = dai.Pipeline()


monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
color = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
imu = pipeline.create(dai.node.IMU)
sync = pipeline.create(dai.node.Sync)

color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
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

imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 400)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
imu.out.link(xlinkOut.input)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

color.setCamera("color")

sync.setSyncThreshold(timedelta(milliseconds=50))

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


coordinates_3d = [0,0,0]

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
                    boxes_xy = result.boxes.xyxy[0]

                    center_x = int((boxes_xy[0]+boxes_xy[2])/2)
                    center_y = int((boxes_xy[1]+boxes_xy[3])/2)
                    
                    top_left_x = int(boxes_xy[0])
                    # print(center_x, center_y)
                    if 400>center_y>40:
                        topLeft = dai.Point2f((boxes_xy[0])/640, (boxes_xy[1]-40)/400)# (960,540))[ 30:510, 160:800]
                        bottomRight = dai.Point2f((boxes_xy[2])/640,(boxes_xy[3]-40)/400)
                    
                    cur_angle = np.array(angles) - np.array(angles_default)
                    frame["video"] = cv2.circle(frame["video"], (center_x,center_y), 10, (255, 0, 0) , 2) 
                    
                    #roll pitch yaw
                    Global_xyz = tramform2base(coordinates_3d[0],coordinates_3d[1],coordinates_3d[2],cur_angle)
                    
                    print(f"Global X: {Global_xyz[0]} mm, Y: {Global_xyz[1]} mm, Z: {Global_xyz[2]} mm")

                    CvCmder.CvCmd_Heartbeat(gimbal_pitch_target=Global_xyz[1], gimbal_yaw_target=Global_xyz[2], chassis_speed_x=0, chassis_speed_y=0)
                    break 
         
        cv2.imshow("video", frame["video"])
        cv2.imshow("depth", frame["disparity"])

    if cv2.waitKey(1) == ord('q'):
        break

#kill the camera thread    
thread1.join()