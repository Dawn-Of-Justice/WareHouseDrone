#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import cv2.aruco as aruco

class Arena(Node):

    def __init__(self):
        super().__init__('aruco')
        self.corners_final = {}    
        self.width = 1000
        self.height = 1000
        self.grid_size = 17
        self.current_transform = None
        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.br = CvBridge()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_transform is not None:
            # Get current count of points from file
            try:
                with open("pixel_values.txt", "r") as f:
                    count = len(f.readlines())
            except FileNotFoundError:
                count = 0
                
            # Generate point name (A1, B1, etc.)
            row = (count // 17) + 1
            col = chr(65 + (count % 17))
            point_name = f"{col}{row}"
                
            # Save to file
            with open("pixel_values.txt", "a") as f:
                f.write(f"{point_name}: {(x,y)}\n")

    def image_callback(self, img):
        self.get_logger().info('Received image')
        current_frame = self.br.imgmsg_to_cv2(img, "bgr8")
        self.identification(current_frame)

    def identification(self, current_frame):
        frame = current_frame
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        markerCorners, markerIds, rejectedImgPoints = detector.detectMarkers(frame)

        if markerIds is not None:
            self.get_logger().info(f'Detected markers: {markerIds}')
        else:
            self.get_logger().info('No markers detected')

        required_marker_ids = [80, 85, 90, 95]

        while set(required_marker_ids).issubset(self.corners_final.keys()) == False:
            for i in range(len(markerIds)):
                if markerIds[i][0] in required_marker_ids and markerIds[i][0] not in self.corners_final:
                    self.corners_final[markerIds[i][0]] = [markerCorners[i][0].tolist()]
        
        if set(required_marker_ids).issubset(self.corners_final.keys()):
            pts1 = np.float32([self.corners_final[80][0][0], self.corners_final[85][0][1], 
                              self.corners_final[95][0][3], self.corners_final[90][0][2]])
            pts2 = np.float32([[0,0],[self.width,0],[0,self.height],[self.width,self.height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.current_transform = cv2.warpPerspective(frame, matrix, (self.width, self.height))
            
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('image', self.mouse_callback)
            cv2.imshow('image', self.current_transform)          
            cv2.resizeWindow('image', 900, 900)
            cv2.waitKey(1)
        else:
            self.get_logger().info('Required markers not yet detected')

def main(args=None):
    rclpy.init(args=args)
    arena = Arena()

    try:
        rclpy.spin(arena)
    except KeyboardInterrupt:
        pass

    arena.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()