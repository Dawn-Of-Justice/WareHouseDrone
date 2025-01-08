#!/usr/bin/env python3
"""
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
# Filename:         WD_1284_pico_client.py
# Functions:        __init__, expand_path, is_goal_point, send_goal, goal_response_callback, get_result_callback, feedback_callback, send_request, receive_goals, main
# Global variables: None
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints

class WayPointClient(Node):

    def __init__(self):
        """
        Purpose:
        ---
        Initialize the WayPointClient node, setup clients, and prepare for communication.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        """
        super().__init__("waypoint_client")
        self.goals = []
        self.intermediate_goals = []  # Store expanded path with intermediate points
        self.goal_index = 0
        self.action_client = ActionClient(self, NavToWaypoint, "waypoint_navigation")

        self.cli = self.create_client(GetWaypoints, "waypoints")
        self.current_position = [0.0, 0.0, 26.0]  # Track current position

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")

        self.req = GetWaypoints.Request()

    def expand_path(self, waypoints):
        """
        Purpose:
        ---
        Expand waypoints into intermediate points moving only along axes.

        Input Arguments:
        ---
        waypoints : list
            List of waypoints to expand.

        Returns:
        ---
        expanded : list
            List of expanded waypoints with intermediate points.
        """

        expanded = []
        result = []
        current_pos = self.current_position

        for waypoint in waypoints:
            # First add intermediate point that moves only in X
            intermediate = [waypoint[0], current_pos[1], waypoint[2]]
            expanded.append(intermediate)

            # Then add the final point that moves in Y
            expanded.append(waypoint)

            # Update current position for next iteration
            current_pos = waypoint

        result.append(expanded[0])
        for i in range(1,len(expanded)):
            if result[-1] == expanded[i]:
                continue
            else:
                result.append(expanded[i])

        return result

    def is_goal_point(self, waypoint):
        """
        Purpose:
        ---
        Check if the given waypoint is a final goal point (not intermediate).

        Input Arguments:
        ---
        waypoint : list
            The waypoint to check.

        Returns:
        ---
        bool
            True if waypoint is a goal point, False otherwise.
        """
        return waypoint in self.goals

    def send_goal(self, waypoint):
        """
        Purpose:
        ---
        Send a goal to the action server.

        Input Arguments:
        ---
        waypoint : list
            The waypoint to send as a goal.

        Returns:
        ---
        None
        """
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint.position.x = waypoint[0]
        goal_msg.waypoint.position.y = waypoint[1]
        goal_msg.waypoint.position.z = waypoint[2]

        # Check if this is a goal point or intermediate point
        goal_msg.is_goal_point = self.is_goal_point(waypoint)

        # Log the point type
        point_type = "goal" if goal_msg.is_goal_point else "intermediate"
        self.get_logger().info(f"Sending {point_type} point: {waypoint}")

        # Wait for action server
        self.get_logger().info("Waiting for action server...")
        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Action server not available, waiting...")

        self.get_logger().info(f"Sending goal: {waypoint}")
        self.send_goal_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Purpose:
        ---
        Handle the response from the action server for a goal.

        Input Arguments:
        ---
        future : Future
            The future object representing the response.

        Returns:
        ---
        None
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Purpose:
        ---
        Handle the result of a goal execution.

        Input Arguments:
        ---
        future : Future
            The future object representing the result.

        Returns:
        ---
        None
        """
        result = future.result().result
        self.get_logger().info("Result: {0}".format(result.hov_time))

        # Update current position with the reached waypoint
        self.current_position = self.intermediate_goals[self.goal_index]
        self.goal_index += 1

        if self.goal_index < len(self.intermediate_goals):
            self.send_goal(self.intermediate_goals[self.goal_index])
        else:
            self.get_logger().info("All waypoints have been reached successfully")

    def feedback_callback(self, feedback_msg):
        """
        Purpose:
        ---
        Process feedback from the action server.

        Input Arguments:
        ---
        feedback_msg : Feedback
            Feedback message from the server.

        Returns:
        ---
        None
        """
        feedback = feedback_msg.feedback
        x = feedback.current_waypoint.pose.position.x
        y = feedback.current_waypoint.pose.position.y
        z = feedback.current_waypoint.pose.position.z
        t = feedback.current_waypoint.header.stamp.sec

    def send_request(self):
        """
        Purpose:
        ---
        Send a request to the service for waypoints.

        Input Arguments:
        ---
        None

        Returns:
        ---
        Future
            Future object representing the service call.
        """
        self.req.get_waypoints = True
        self.get_logger().info("Sending service request...")
        return self.cli.call_async(self.req)

    def receive_goals(self):
        """
        Purpose:
        ---
        Receive waypoints from the service and start navigation.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        """
        future = self.send_request()
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        self.get_logger().info("Waypoints received by the action client")

        # Store original waypoints
        for pose in response.waypoints.poses:
            waypoints = [pose.position.x, pose.position.y, pose.position.z]
            self.goals.append(waypoints)
            self.get_logger().info(f"Original waypoint: {waypoints}")

        # Expand the path to include intermediate points
        self.intermediate_goals = self.expand_path(self.goals)
        print("Expanded path with intermediate points:")
        for point in self.intermediate_goals:
            print(point)

        # Start navigation with first point
        self.send_goal(self.intermediate_goals[0])


def main(args=None):
    """
    Purpose:
    ---
    Initialize the ROS client library, create the WayPointClient node, and start the waypoint navigation process.

    Input Arguments:
    ---
    args : list, optional
        Command-line arguments passed to the ROS 2 client library (default is None).

    Returns:
    ---
    None
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


if __name__ == "__main__":
    main()