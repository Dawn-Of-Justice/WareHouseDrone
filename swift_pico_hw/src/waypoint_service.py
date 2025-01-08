#!/usr/bin/env python3
"""
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S
# Filename:         WD_1284_waypoint_service.py
# Functions:        __init__, waypoint_callback, main
# Global variables: None
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from waypoint_navigation.srv import GetWaypoints

class WayPoints(Node):

    def __init__(self):
        """
        Purpose:
        ---
        Initialize the WayPoints node, setup services, and prepare for communication.
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        """
        super().__init__('waypoints_service')
        self.srv = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback)
        self.waypoints = [[2.0, 2.0, 26.0], [2.0, -2.0, 26.0], [-2.0, -2.0, 26.0], [-2.0, 2.0, 26.0], [1.0, 1.0, 26.0]]

    
    def waypoint_callback(self, request, response):
        """
        Purpose:
        ---
        Callback function for the GetWaypoints service. Sends the waypoints to the client.
        
        Input Arguments:
        ---
        request : GetWaypoints.Request
            Request object containing the request data.
        response : GetWaypoints.Response
            Response object containing the response data.
        
        Returns:
        ---
        response : GetWaypoints.Response
            Response object containing the waypoints.
            
        Example call:
        ---
        response = self.waypoint_callback
        
        """

        if request.get_waypoints == True :
            response.waypoints.poses = [Pose() for _ in range(len(self.waypoints))]
            for i in range(len(self.waypoints)):
                response.waypoints.poses[i].position.x = self.waypoints[i][0]
                response.waypoints.poses[i].position.y = self.waypoints[i][1]
                response.waypoints.poses[i].position.z = self.waypoints[i][2]
            self.get_logger().info("Incoming request for Waypoints")
            return response

        else:
            self.get_logger().info("Request rejected")

def main():
    """
    Purpose:
    ---
    Initialize the WayPoints node and start the service.
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    None
    
    Example call:
    ---
    main()
    """
    
    rclpy.init()
    waypoints = WayPoints()

    try:
        rclpy.spin(waypoints)
    except KeyboardInterrupt:
        waypoints.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoints.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
        

        