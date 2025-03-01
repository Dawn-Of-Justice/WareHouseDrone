#!/usr/bin/env python3

"""
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
# Filename:         WD_1284_pico_client_5.py
# Functions:        __init__, send_goal, goal_response_callback, get_result_callback, handle_receive_goals_response, feedback_callback, process_nest_goal, send_request, receive_goals, main
# Global variables: None
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints
from collections import deque
from rclpy.callback_groups import ReentrantCallbackGroup


class WayPointClient(Node):

    def __init__(self):
        """
        Purpose:
        ---
        Initialize the Client node

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        WayPointClient()
        """
        super().__init__("waypoint_client")
        self.goals = deque()
        self.callback_group = ReentrantCallbackGroup()
        self.action_client = ActionClient(
            self,
            NavToWaypoint,
            "waypoint_navigation",
            callback_group=self.callback_group,
        )
        self.is_executing = False
        self.cli = self.create_client(
            GetWaypoints, "waypoints", callback_group=self.callback_group
        )
        
        # Used to keep track of how many goals to visit
        self.current_package = 0
        self.package_index = [2,3]  # First package index is 2

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")

        self.req = GetWaypoints.Request()

        
    def send_goal(self, waypoint):
        """
        Purpose:
        ---
        Sends a navigation goal to the action server

        Input Arguments:
        ---
        `waypoint` :  [ list ]
            point to be sent

        Returns:
        ---
        None

        Example call:
        ---
        send_goal([1,2,23])
        """
        if self.is_executing:
            self.goals.append(waypoint)
            self.get_logger().info(f"Added waypoint to queue: {waypoint}")
            return

        self.is_executing = True
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint.position.x = waypoint[0]
        goal_msg.waypoint.position.y = waypoint[1]
        goal_msg.waypoint.position.z = waypoint[2]
        goal_msg.is_goal_point = len(self.goals) == 0
        if len(self.goals) == 0:
            self.current_package += 1
            self.get_logger().info(f"Current package: {self.current_package}")
            
        if len(self.goals) == 0 and self.current_package == (len(self.package_index)+1):
            goal_msg.avada_kedavra = True
        else:
            goal_msg.avada_kedavra = False
        print(
            f"Sending goal: x={waypoint[0]}, y={waypoint[1]}, z={waypoint[2]}, is_goal={goal_msg.is_goal_point}")

        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Action server not available, waiting...")


        send_goal_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Purpose:
        ---
        Callback after the server reaches the goal

        Input Arguments:
        ---
        `future` :  [ rclpy.task.Future ]
            The future object representing the result of the goal request.
Leave a small buffer for disarm
        Returns:
        ---
        None

        Example call:
        ---
        send_goal_future.add_done_callback(self.goal_response_callback)
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            self.is_executing = False
            self.process_next_goal()
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self.get_logger().info("Waiting for result...")
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Purpose:
        ---
        Callback function to receive the result of drone reaching the goal

        Input Arguments:
        ---
        `future` :  [ rclpy.task.Future ]
            The future object representing the result of the goal request.

        Returns:
        ---
        None

        Example call:
        ---
        self._get_result_future.add_done_callback(self.get_result_callback)
        """
        result = future.result().result
        self.get_logger().info("Result: {0}".format(result.hov_time))
        self.is_executing = False

        if result and len(self.goals) == 0:
            print("Requesting new sub goals from result callback")
            future = self.send_request()
            future.add_done_callback(self.handle_receive_goals_response)
            return

        self.process_next_goal()

    def handle_receive_goals_response(self, future):
        """
        Purpose:
        ---
        Callback function to process the path after receiving it from the path planning service

        Input Arguments:
        ---
        `future` :  [ rclpy.task.Future ]
            The future object representing the result of the request.

        Returns:
        ---
        None

        Example call:
        ---
        future.add_done_callback(self.handle_receive_goals_response)
        """
        try:
            response = future.result()
            self.get_logger().info("Waypoints received by the action client")

            # Clear existing goals before adding new ones
            self.goals.clear()
            if not response.waypoints.poses:
                self.get_logger().info("Landing sequence")
                    
            for pose in response.waypoints.poses:
                waypoint = [pose.position.x, pose.position.y, pose.position.z]
                self.goals.append(waypoint)
                self.get_logger().info(f"Waypoint received: {waypoint}")

            self.get_logger().info(f"Waypoint received: {waypoint}")
            
            # Process the next goal if we're not currently executing
            if not self.is_executing:
                self.process_next_goal()

        except Exception as e:
            self.get_logger().error(f"Error processing new waypoints: {str(e)}")
            self.is_executing = False
            self.process_next_goal()

    def feedback_callback(self, feedback_msg):
        """
        Purpose:
        ---
        Handles the feedback received from the action server during the execution of a goal. It extracts the current waypoint's position from the feedback message.

        Input Arguments:
        ---
        `feedback_msg` :  [ < type of 1st input argument > ]
            < one-line description of 1st input argument >

        Returns:
        ---
        None

        Example call:
        ---
        feedback_callback(feedback_msg)
        """
        feedback = feedback_msg.feedback
        x = feedback.current_waypoint.pose.position.x
        y = feedback.current_waypoint.pose.position.y
        z = feedback.current_waypoint.pose.position.z
        t = feedback.current_waypoint.header.stamp.sec


    def process_next_goal(self):
        """
        Purpose:
        ---
        Processes the next goal in the queue by sending it to the action server.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        process_next_goal()
        """
        if self.goals:
            wayp = self.goals.popleft()
            self.send_goal(wayp)
        else:
            self.get_logger().info("All waypoints have been reached successfully")
            self.is_executing = False

    def send_request(self):
        """
        Purpose:
        ---
        Sends a service request to retrieve waypoints.

        Input Arguments:
        ---
        None

        Returns:
        ---
        `future` :  [ rclpy.task.Future ]
            A future object that will hold the result of the asynchronous service call.

        Example call:
        ---
        send_request()
        """
        self.req.get_waypoints = True
        self.get_logger().info("Sending service request...")
        return self.cli.call_async(self.req)

    def receive_goals(self):
        """
        Purpose:
        ---
        Requests path from the service and sets up a callback to handle the response.

        Input Arguments:
        ---Leave a small buffer for disarm
        None

        Returns:
        ---
        None

        Example call:
        ---
        receive_goals()
        """
        print("Requesting initial goals")
        future = self.send_request()
        future.add_done_callback(self.handle_receive_goals_response)


def main(args=None):
    """
    Purpose:
    ---
    Main function to initialize the node and run the client

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
    rclpy.init(args=args)

    waypoint_client = WayPointClient()
    waypoint_client.receive_goals()

    try:
        rclpy.spin(waypoint_client)
    except KeyboardInterrupt:
        waypoint_client.get_logger().info("KeyboardInterrupt, shutting down.\n")
    finally:
        waypoint_client.destroy_node()
        rclpy.shutdown()

    rclpy.shutdown()


if __name__ == "__main__":
    main()