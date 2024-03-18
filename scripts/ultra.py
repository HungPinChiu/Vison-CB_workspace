#!/usr/bin/env python3
## Vision
from ultralytics import YOLO
import pyrealsense2 as rs

## ROS
import tf
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int8MultiArray

#Self-defined msg
from yolo.srv import RD_signal, RD_signalResponse

## Other tools
import numpy as np
from cv_bridge import CvBridge
from threading import Thread, Event

# Define variables
DEVICE_SERIAL = "215222079970" # Realsense device serial number
XOFFSET = 0.0 # x distance with CB center
ZOFFSET = 0.0 # Z distance with CB center
THETA_HORI = 0 # camera rotation angle within CB in degree
THETA_VERTI = 0 # camera depression angle within CB in degree
WIN_WIDTH, WIN_HEIGHT = 640, 480 # camera resolution
VERBOSE = True # YOLO verbose (showing detection ouput)
WAITING_TIME = 0 # waiting time for first publish

six_region_map = {
    # "Region": [x1, x2, y1, y2, z1, z2]
    "1": [300, 380, 300, 340, 1.1, 1.3],
    "2": [0, 0, 0, 0, 0, 0],
    "3": [50, 90, 120, 160, 1.6, 1.8],
    "4": [220, 270, 40, 80, 1.8, 2.0],
    "5": [400, 450, 50, 100, 1.7, 1.9],
    "6": [470, 510, 180, 240, 1.4, 1.6]
}

class Node:
    def __init__(self):
        rospy.init_node('CB_Server')

        ### YOLO model ###
        self.model = YOLO("src/yolo/weight/best.pt")
        #self.model.fuse() # Fuse for speed
        self.results_img = None

        ### Publisher ###
        # CB detection topic
        self.pub = rospy.Publisher('/robot/objects/global_info', Int8MultiArray, queue_size=10)
        self.six_plant_info = Int8MultiArray()
        self.six_plant_info.data = [0] * 6
        # GUI Publisher
        self.yolo_result_pub = rospy.Publisher('/robot/objects/yolo_result', Image, queue_size=10)
        # Ready signal service
        self.service = rospy.Service('/robot/startup/ready_signal', RD_signal, self.ready_callback)
        
        ### Other tools ###
        # CvBridge
        self.bridge = CvBridge()
        # tf_listener
        self.tf_listener = tf.TransformListener()
        self.tf_listener.waitForTransform("map", "realsense_camera", rospy.Time(0), rospy.Duration(10.0))
        self.camera_point, self.world_point = PointStamped()
        self.camera_point.header.frame_id = "realsense_camera"
        self.camera_point.header.stamp = rospy.Time.now()

        rospy.loginfo("CB waiting for Ready Signal...")
        self.publish_thread = None
        self.first_publish_event = Event()

    def ready_callback(self, request) -> RD_signalResponse:
        ready = request.ready
        if ready:
            rospy.loginfo("Received ready signal. Starting CB detection...")
            if self.publish_thread is None or self.publish_thread.is_alive() is False:
                self.first_publish_event.clear()

                ###### YOLO STARTS HERE ######
                self.publish_thread = Thread(target=yolo) 
                self.publish_thread.start()

                # Wait for the first publish to complete
                self.first_publish_event.wait()

            success = True
        else:
            rospy.logwarn("Received ready signal is False. Aborting process...")
            success = False

        return RD_signalResponse(success)

def yolo():
    while not rospy.is_shutdown():
        color_img, depth_img = realsense_class.wait_for_frames()
        np_color_img = np.asanyarray(color_img.get_data(), dtype = np.uint8)

        #YOLO detection
        results = self.model(source = np_color_img, verbose = VERBOSE)

        for object in results:
            results_img = object.plot()
            self.yolo_result_pub.publish(bridge.cv2_to_imgmsg(results_img, encoding="bgr8")) 
            boxes = object.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                Xframe, Yframe = int((x1 + x2) / 2), int((y1 + y2) / 2)
                depth = depth_img.get_distance(Xframe, Yframe)
                # Transform coordinates
                transform_coordinates(Xframe, Yframe, depth)
                # print(f"X: {x}, Y: {y}, Z: {depth}")
                
                # Check 6 plant info
                six_plant_info_check(world_point.point.x, world_point.point.y, world_point.point.z)

        # From first publish check
        if first_publish_event.is_set() is False:
            rospy.sleep(WAITING_TIME)
            rospy.loginfo("Processing YOLO...")
            first_publish_event.set()

        # Publish 6 plant info
        pub.publish(six_plant_info)
        six_plant_info.data = [0] * 6

def transform_coordinates(Xframe, Yframe, depth):
    # Transform to camera coordinates
    camera_point = PointStamped()
    camera_point.point.x = depth*(Xframe - realsense_class.intr.ppx) / realsense_class.intr.fx
    camera_point.point.y = depth*(Yframe - realsense_class.intr.ppy) / realsense_class.intr.fy
    camera_point.point.z = depth

    # Transform to world coordinates
    world_point = tf_listener.transformPoint("map", camera_point) 

def six_plant_info_check(x, y, depth):
    # if match the region, set the corresponding value to 1
    for region, value in six_region_map.items():
        if (value[0] < x < value[1]) and (value[2] < y < value[3]) and (value[4] < depth < value[5]):
            six_plant_info.data[int(region)-1] = 1

class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        # Camera configuration
        self.config = rs.config()
        self.config.enable_device(DEVICE_SERIAL)
        self.config.enable_stream(rs.stream.color, WIN_WIDTH, WIN_HEIGHT, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, WIN_WIDTH, WIN_HEIGHT, rs.format.z16, 30)
        # Streaming configuration
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def wait_for_frames(self):
        aligned_frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        return color_frame, depth_frame
    
if __name__ == '__main__':
    try:
        vision_node = Node()
        realsense_class = RealsenseCamera()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
