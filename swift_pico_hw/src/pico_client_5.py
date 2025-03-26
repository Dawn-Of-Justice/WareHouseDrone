#!/usr/bin/env python3
"""
* Team Id : 1284
* Author List : Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
* Filename: WD_1284_pico_client_5.py
* Theme: WareHouse Drone
* Functions: __init__, send_goal, goal_response_callback, get_result_callback, handle_receive_goals_response, feedback_callback, process_next_goal, send_request, receive_goals, main
* Global Variables: None
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints
from collections import deque
from rclpy.callback_groups import ReentrantCallbackGroup


"""
Class Name: WayPointClient
Purpose: Manages the client node for sending and receiving waypoints in a warehouse drone system.
"""
class WayPointClient(Node):

    """
    * Function Name: __init__
    * Input: None
    * Output: None
    * Logic: Initializes the ROS node, sets up action clients, service clients, and prepares for communication.
    * Example Call: waypoint_client = WayPointClient()
    """
    def __init__(self):
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
        self.package_index = [1]  # First package index is 2

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")

        self.req = GetWaypoints.Request()

    """
    * Function Name: set_package_list
    * Input: msg (String) - Message containing package index.
    * Output: None
    * Logic: Sets the package index based on received message.
    * Example Call: set_package_list(msg)
    """
    def set_package_list(self, msg):
        self.package_index = [int(msg.data)]
        
    """
    * Function Name: send_goal
    * Input: waypoint (list) - The waypoint to be sent as a goal.
    * Output: None
    * Logic: Sends a navigation goal to the action server.
    * Example Call: self.send_goal([1,2,23])
    """
    def send_goal(self, waypoint):
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

    """
    * Function Name: goal_response_callback
    * Input: future (Future) - Future object representing the response.
    * Output: None
    * Logic: Handles the server response after sending a goal.
    * Example Call: Called internally by send_goal.
    """
    def goal_response_callback(self, future):
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

    """
    * Function Name: get_result_callback
    * Input: future (Future) - Future object representing the result.
    * Output: None
    * Logic: Processes the result after reaching a goal and proceeds to the next goal if available.
    * Example Call: Called internally by goal_response_callback.
    """
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("Result: {0}".format(result.hov_time))
        self.is_executing = False

        if result and len(self.goals) == 0:
            print("Requesting new sub goals from result callback")
            future = self.send_request()
            future.add_done_callback(self.handle_receive_goals_response)
            return

        self.process_next_goal()

    """
    * Function Name: handle_receive_goals_response
    * Input: future (Future) - Future object representing the request result.
    * Output: None
    * Logic: Processes received waypoints and initiates execution.
    * Example Call: future.add_done_callback(self.handle_receive_goals_response)
    """
    def handle_receive_goals_response(self, future):
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

    """
    * Function Name: feedback_callback
    * Input: feedback_msg - Feedback message from the action server.
    * Output: None
    * Logic: Extracts and logs the current waypoint's position.
    * Example Call: feedback_callback(feedback_msg)
    """
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        x = feedback.current_waypoint.pose.position.x
        y = feedback.current_waypoint.pose.position.y
        z = feedback.current_waypoint.pose.position.z
        t = feedback.current_waypoint.header.stamp.sec


    """
    * Function Name: process_next_goal
    * Input: None
    * Output: None
    * Logic: Sends the next goal from the queue if available.
    * Example Call: self.process_next_goal()
    """
    def process_next_goal(self):
        if self.goals:
            wayp = self.goals.popleft()
            self.send_goal(wayp)
        else:
            self.get_logger().info("All waypoints have been reached successfully")
            self.is_executing = False

    """
    * Function Name: send_request
    * Input: None
    * Output: Future - Future object representing the service call.
    * Logic: Sends a request to retrieve waypoints.
    * Example Call: future = self.send_request()
    """
    def send_request(self):
        self.req.get_waypoints = True
        self.get_logger().info("Sending service request...")
        return self.cli.call_async(self.req)

    """
    * Function Name: receive_goals
    * Input: None
    * Output: None
    * Logic: Requests path from the service and sets up a callback to handle the response.
    * Example Call: self.receive_goals()
    """
    def receive_goals(self):
        print("Requesting initial goals")
        future = self.send_request()
        future.add_done_callback(self.handle_receive_goals_response)


"""
* Function Name: main
* Input: args (list, optional) - Command-line arguments for ROS 2.
* Output: None
* Logic: Initializes ROS client and starts the WayPointClient node.
* Example Call: main()
"""
def main(args=None):
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