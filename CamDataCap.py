import depthai as dai
from datetime import datetime, timedelta
import numpy as np
import cv2
import CvHelper   

class depth_camera:
    
    topLeft = dai.Point2f(0.45, 0.45)
    bottomRight = dai.Point2f(0.55, 0.55)
    pipeline = dai.Pipeline()
    calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
    coordinates_3d = [0,0,0]
    angles_default = [0,0,0]
    angles_min =[0,0,0]
    colorfps = 60
    cur_angle = [0,0,0]
    cam_started = False
    depth_on = False

    def __init__(self):
        
        self.frame = {"video": None, "disparity": None}
        self.pipeline = dai.Pipeline()

        #create pipeline components

        
        
        #color camera
        color = self.pipeline.create(dai.node.ColorCamera)
        #Config color camera
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setFps(self.colorfps)
        color.setCamera("color")

        manip = self.pipeline.createImageManip()


        #imu
        imu = self.pipeline.create(dai.node.IMU)



        if self.depth_on:

            monoLeft = self.pipeline.create(dai.node.MonoCamera)
            monoRight = self.pipeline.create(dai.node.MonoCamera)

            stereo = self.pipeline.create(dai.node.StereoDepth)
            spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)

            #Add for depth video sync
            xoutDepth = self.pipeline.create(dai.node.XLinkOut)
            xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
            xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)

            xoutDepth.setStreamName("depth")
            xoutSpatialData.setStreamName("spatialData")
            xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

            #Set up mono camera resolution on left and right
            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoLeft.setCamera("left")
            monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoRight.setCamera("right")

            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)

            #config roi
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 100
            config.depthThresholds.upperThreshold = 10000

            config.roi = dai.Rect(self.topLeft, self.bottomRight)
            
            #Spatial location calculator
            spatialLocationCalculator.inputConfig.setWaitForMessage(False)
            spatialLocationCalculator.initialConfig.addROI(config)

            monoLeft.out.link(stereo.left)
            monoRight.out.link(stereo.right)
            spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
            stereo.depth.link(spatialLocationCalculator.inputDepth)

            spatialLocationCalculator.out.link(xoutSpatialData.input)
            xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


        #sync
        sync = self.pipeline.create(dai.node.Sync)

        #Create LinkOut nodes
        xoutGrp = self.pipeline.create(dai.node.XLinkOut)
        xoutGrp.setStreamName("xout")

        #Add for IMU Xlink
        xlinkOut = self.pipeline.create(dai.node.XLinkOut)
        xlinkOut.setStreamName("imu")
    

        #Enable IMU
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 120)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

       

        

        #setting up sync timeout
        sync.setSyncThreshold(timedelta(milliseconds=10))
        sync.setSyncAttempts(-1)


        imu.out.link(sync.inputs["imu"])
        color.video.link(sync.inputs["video"])

        sync.out.link(xoutGrp.input)

        if self.depth_on:
            stereo.depth.link(sync.inputs["disparity"])
            self.disparityMultiplier = 255.0 / stereo.initialConfig.getMaxDisparity()
        
    def start(self):
        # Pipeline is defined, now we can connect to the device
        with dai.Device(self.pipeline) as device:

            init_state = True
            queue = device.getOutputQueue("xout", 3, True)
            
            baseTs = None

            if self.depth_on:
                depthQueue = device.getOutputQueue(name="depth", maxSize=2, blocking=False)
                spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=2, blocking=False)
                spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

            while True:
                #get RGB data
                msgGrp = queue.get()

                #get IMU data
                imuData = msgGrp['imu']
                colorData = msgGrp['video']

                #unpack RGB data
                

                if self.depth_on:
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
                        self.coordinates_3d = [int(depthData.spatialCoordinates.z),int(depthData.spatialCoordinates.x) ,-int(depthData.spatialCoordinates.y)]    



                frame_get = colorData.getCvFrame()
                self.frame["video"] = cv2.rotate(cv2.resize(frame_get, (640,360)),cv2.ROTATE_180)

                if self.depth_on:
                    self.frame["disparity"] = (frame_get * self.disparityMultiplier).astype(np.uint8)
                    self.frame["disparity"]= cv2.applyColorMap(self.frame["disparity"], cv2.COLORMAP_JET)

                #unpack IMU data
                latestRotVec = imuData.packets[-1].rotationVector

               

                # set precision
                # imuF = "{:.06f}"
                # tsF  = "{:.03f}"

                #i : yaw, j : roll, k : pitch
                self.cur_angle = CvHelper.euler_from_quaternion(latestRotVec.i,latestRotVec.j,latestRotVec.k,latestRotVec.real)
                    
                if init_state:
                    self.angles_default = self.cur_angle
                    self.angles_min[2] = self.angles_default[2]
                    init_state = False

                # config = dai.SpatialLocationCalculatorConfigData()    
                # config.roi = dai.Rect(self.topLeft, self.bottomRight)
                # config.calculationAlgorithm = self.calculationAlgorithm
                # cfg = dai.SpatialLocationCalculatorConfig()
                # cfg.addROI(config)
                # spatialCalcConfigInQueue.send(cfg)

    def get_frame(self):
        return self.frame
    
    def get_coordinates(self):
        return self.coordinates_3d
    
    def get_default_angles(self):
        return self.angles_default
    
    def set_ROI(self, topLeft, bottomRight):
        self.topLeft = dai.Point2f(topLeft[0], topLeft[1])
        self.bottomRight = dai.Point2f(bottomRight[0], bottomRight[1])

    def close(self):
        self.device.close()

