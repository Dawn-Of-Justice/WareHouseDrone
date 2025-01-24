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
        self.all_markers_detected = False
        self.bitmap = None
        self.detected m
        
        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.br = CvBridge()
        
        # Add debug window initialization
        cv2.namedWindow('original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('transformed', cv2.WINDOW_NORMAL)

    def image_callback(self, img):
        try:
            current_frame = self.br.imgmsg_to_cv2(img, "bgr8")
            # Show original frame for debugging
            cv2.imshow('original', current_frame)
            
            if not self.all_markers_detected:
                self.identification(current_frame)
            
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}')

    def identification(self, current_frame):
        try:
            frame = current_frame.copy()  # Create a copy to avoid modifying original
            self.corners_final = {}

            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
            parameters = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            markerCorners, markerIds, rejectedImgPoints = detector.detectMarkers(frame)

            # Draw detected markers for debugging
            if markerIds is not None:
                frame = aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
                cv2.imshow('original', frame)
                
                self.get_logger().info(f'Detected markers: {markerIds}')
                required_marker_ids = [80, 85, 90, 95]

                # Add newly detected markers to corners_final
                for i in range(len(markerIds)):
                    marker_id = markerIds[i][0]
                    if marker_id in required_marker_ids and marker_id not in self.corners_final:
                        corners = markerCorners[i][0].tolist()
                        self.corners_final[marker_id] = [corners]
                        self.get_logger().info(f'Marker {marker_id} corners: {corners}')

                # Check if all required markers are detected
                if set(required_marker_ids).issubset(self.corners_final.keys()):
                    pts1 = np.float32([
                        self.corners_final[80][0][0],  # Top-left
                        self.corners_final[85][0][1],  # Top-right
                        self.corners_final[95][0][3],  # Bottom-left
                        self.corners_final[90][0][2]   # Bottom-right
                    ])
                    
                    
                    pts2 = np.float32([
                        [0, 0],
                        [self.width, 0],
                        [0, self.height],
                        [self.width, self.height]
                    ])
                    
                    # Check if points are valid
                    if np.all(np.isfinite(pts1)):
                        matrix = cv2.getPerspectiveTransform(pts1, pts2)
                        perspective_transform_output = cv2.warpPerspective(
                            frame, matrix, (self.width, self.height)
                        )
                        
                        self.get_logger().info('Perspective transform completed')
                        cv2.imshow('transformed', perspective_transform_output)
                        cv2.resizeWindow('transformed', 900, 900)
                        self.all_markers_detected = True
                    else:
                        self.get_logger().error('Invalid source points detected')
                else:
                    missing_markers = set(required_marker_ids) - set(self.corners_final.keys())
                    self.get_logger().info(f'Waiting for markers: {missing_markers}')
            else:
                self.get_logger().debug('No markers detected in current frame')
            
            self.find_obstacles(perspective_transform_output)  
            if self.bitmap is not None:
                cv2.imwrite('2d_bit_map.png', self.bitmap)
                self.text_file() 
        except Exception as e:
            self.get_logger().error(f'Error in identification: {str(e)}')
    
    def apply_perspective_transform(self, image, corners):
        try:
            if len(corners) != 4:
                self.get_logger().error("Error: Exactly 4 ArUco markers are required for perspective transform")
                return None

            h, w = image.shape[:2]
            
            def corner_distance(point, corner):
                return np.sqrt((point[0] - corner[0])**2 + (point[1] - corner[1])**2)

            top_left = max(corners, key=lambda c: corner_distance(c[0][0], (0, 0)))
            top_right = max(corners, key=lambda c: corner_distance(c[0][0], (w, 0)))
            bottom_right = max(corners, key=lambda c: corner_distance(c[0][0], (w, h)))
            bottom_left = max(corners, key=lambda c: corner_distance(c[0][0], (0, h)))

            source_points = np.array([
                top_left[0][2],
                top_right[0][3],
                bottom_right[0][0],
                bottom_left[0][1]
            ], dtype=np.float32)

            dest_points = np.array([
                [0, 0],
                [self.width - 1, 0],
                [self.width - 1, self.height - 1],
                [0, self.height - 1]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(source_points, dest_points)
            result = cv2.warpPerspective(image, matrix, (self.width, self.height))
            result = cv2.rotate(result, cv2.ROTATE_180)
            # Get and apply the perspective transform
            
            return result

        except Exception as e:
            self.get_logger().error(f'Error in perspective transform: {str(e)}')
            return None
        
    def find_obstacles(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding with more moderate parameters
            binary = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 
                363,  # Reduced block size for better detail
                1     # Slightly increased C for better contrast
            )

            # Create a white background bitmap
            self.bitmap = np.full((self.height, self.width), 255, dtype=np.uint8)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            self.obstacles = 0
            self.total_area = 0
            
            min_area = 100  # Minimum area threshold
            border_width = 5  # Border width in pixels
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    self.obstacles += 1
                    self.total_area += area
                    x, y, w, h = cv2.boundingRect(contour)
                    x = max(0, x - 1000)
                    y = max(0, y - 1000)
                    w = min(self.width, w + 1000)
                    h = min(self.height, h + 1000)                    
                    # Add border and fill
                    cv2.drawContours(self.bitmap, [contour], -1, 0, border_width)
                    cv2.drawContours(self.bitmap, [contour], -1, 0, -1)

            self.get_logger().info(f"Detected {self.obstacles} obstacles with total area {self.total_area}")

        except Exception as e:
            self.get_logger().error(f'Error in obstacle detection: {str(e)}')
            self.bitmap = None

    def text_file(self):
        try:
            with open("obstacles.txt", "w") as file:
                file.write(f"Aruco ID: {self.detected_markers}\n")
                file.write(f"Obstacles: {self.obstacles}\n")
                file.write(f"Area: {self.total_area}\n")
        except Exception as e:
            self.get_logger().error(f'Error writing to file: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    arena = Arena()
    
    try:
        rclpy.spin(arena)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        arena.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()