#!/usr/bin/env python3

"""
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
# Filename:         WD_1284_pico_server_2b.py
# Functions:        __init__, butter_lowpass, filter_throttle, disarm, arm, whycon_callback, altitude_set_pid, roll_set_pid, pitch_set_pid, yaw_set_pid, pid, execute_callback, is_drone_in_sphere, main
# Global variables: None
"""

import math
from tf_transformations import euler_from_quaternion
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from waypoint_navigation.action import NavToWaypoint
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PIDTune, PIDError
from nav_msgs.msg import Odometry
from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np


class ThrottleFilter:
    def __init__(self, cutoff, fs, order=5):
        """

        purpose:
        ---
        Initialize the Butterworth filter

        Input Arguments:
        ---
        `cutoff` : [ int ]
            The cutoff frequency of the filter
        `fs` : [ int ]
            The sampling rate of the filter
        `order` : [ int ]
            The order of the filter

        Returns:
        ---
        None

        Example call:
        ---
        ThrottleFilter(2, 1/0.020, 5)

        """
        self.cutoff = cutoff  # cutoff frequency of the filter
        self.fs = fs  # sampling rate
        self.order = order  # order of the filter
        self.b, self.a = self.butter_lowpass()
        self.z = signal.lfilter_zi(self.b, self.a)

    def butter_lowpass(self):
        """

        purpose:
        ---
        Design a Butterworth lowpass filter

        Input Arguments:
        ---
        None

        Returns:
        ---
        `b` : [ array ]
            Numerator coefficients of the filter
        `a` : [ array ]
            Denominator coefficients of the filter

        Example call:
        ---
        butter_lowpass()

        """
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype="low", analog=False)
        return b, a

    def filter_throttle(self, new_throttle):
        """

        purpose:
        ---
        Filter the throttle values using a Butterworth filter

        Input Arguments:
        ---
        `new_throttle` : [ int ]
            The new throttle value to be filtered

        Returns:
        ---
        `filtered_throtle` : [ int ]
            The filtered throttle value

        Example call:
        ---
        filter_throttle(1500)

        """
        new_throttle = np.array([new_throttle])
        filtered_throttle, self.z = lfilter(self.b, self.a, new_throttle, zi=self.z)
        return filtered_throttle[0]


class WayPointServer(Node):

    def __init__(self):
        """
        Purpose:
        ---
        Initialize the node and the controller

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        WayPointServer()
        """

        super().__init__("waypoint_server")

        self.pid_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()

        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.duration = 0

        self.is_stabilized = False
        self.drone_position = [0.0, 0.0, 0.0, 0.0]
        self.setpoint = [0, 0, 27, 0]
        self.dtime = 0

        # Butterworth filter
        self.throttle_filter = ThrottleFilter(
            cutoff=1.8, fs=1 / 0.0125
        )  

        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # [roll, pitch, throttle, yaw]
        self.Kp = [17, 17, 26, 0.3]
        self.Ki = [0.005, 0.005, -0.0025, 0]
        self.Kd = [1400, 1400, 1545, 0.0]

        self.error = [0.0, 0.0, 0.0, 0.0]  # Current errors
        self.prev_error = [0.0, 0.0, 0.0, 0.0]  # Previous errors
        self.error_sum = [0.0, 0.0, 0.0, 0.0]  # Sum of errors (for integral)
        self.d_error = [0.0, 0.0, 0.0, 0.0]  # Change in error (for derivative)

        self.pid_error = PIDError()
        self.sample_time = 0.015

        self.command_pub = self.create_publisher(SwiftMsgs, "/drone_command", 10)
        self.pid_error_pub = self.create_publisher(PIDError, "/pid_error", 10)

        self.create_subscription(PoseArray, "/whycon/poses", self.whycon_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)
        self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, "/yaw_pid", self.yaw_set_pid, 1)
        self.create_subscription(
            Odometry, "/rotors/odometry", self.odometry_callback, 10
        )

        # Action Server
        self._action_server = ActionServer(
            self,
            NavToWaypoint,
            "waypoint_navigation",
            self.execute_callback,
            callback_group=self.action_callback_group,
        )

        self.arm()
        self.timer = self.create_timer(
            self.sample_time, self.pid, callback_group=self.pid_callback_group
        )

    def disarm(self):
        """
        Purpose:
        ---
        Disarm the drone by setting all the commands to 1000

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.disarm()
        """

        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        """
        Purpose:
        ---
        Arm the drone by setting all the commands to 2000

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.arm()
        """

        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.command_pub.publish(self.cmd)

    def whycon_callback(self, msg):
        """
        Purpose:
        ---
        Callback function for whycon poses

        Input Arguments:
        ---
        `msg` : [ PoseArray ]
            The message containing the poses of the drone

        Returns:
        ---
        None

        Example call:
        ---
        self.whycon_callback(msg)
        """

        self.drone_position[0] = msg.poses[0].position.x
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z
        self.dtime = msg.header.stamp.sec

    def altitude_set_pid(self, alt):
        """
        Purpose:
        ---
        Set the PID values for altitude

        Input Arguments:
        ---
        `alt` : [ PIDTune ]
            The message containing the PID values

        Returns:
        ---
        None

        Example call:
        ---
        self.altitude_set_pid(alt)
        """

        self.Kp[2] = alt.kp * 0.03
        self.Ki[2] = alt.ki * 0.001
        self.Kd[2] = alt.kd * 0.1

    def roll_set_pid(self, msg):
        """
        Purpose:
        ---
        Set the PID values for roll

        Input Arguments:
        ---
        `msg` : [ PIDTune ]
            The message containing the PID values

        Returns:
        ---
        None

        Example call:
        ---
        self.roll_set_pid(roll)
        """

        self.Kp[0] = msg.kp * 0.01
        self.Ki[0] = msg.ki * 0.001
        self.Kd[0] = msg.kd * 0.1

    def pitch_set_pid(self, msg):
        """
        Purpose:
        ---
        Set the PID values for pitch

        Input Arguments:
        ---
        'msg' : [ PIDTune ]
            The message containing the PID values

        Returns:
        ---
        None

        Example call:
        ---
        self.pitch_set_pid(pitch)
        """

        self.Kp[1] = msg.kp * 0.01
        self.Ki[1] = msg.ki * 0.001
        self.Kd[1] = msg.kd * 0.1

    def yaw_set_pid(self, msg):
        """
        Purpose:
        ---
        Set the PID values for yaw

        Input Arguments:
        ---
        `msg` : [ PIDTune ]
            The message containing the PID values

        Returns:
        ---
        None

        Example call:
        ---
        self.yaw_set_pid(yaw)
        """

        self.Kp[3] = msg.kp * 0.1
        self.Ki[3] = msg.ki * 0.001
        self.Kd[3] = msg.kd * 0.1

    def odometry_callback(self, msg):
        """
        Purpose:
        ---
        Callback for the odometry message. It processes the incoming odometry data and extracts the orientation information.

        Input Arguments:
        ---
        `msg` : [ nav_msgs.msg.Odometry ]
            The message containing the odometry data, including pose and orientation.

        Returns:
        ---
        None

        Example call:
        ---
        self.odometry_callback(odometry_msg)
        """

        orientation_q = msg.pose.pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)
        self.drone_position[3] = self.yaw_deg

    def pid(self):
        """
        Purpose:
        ---
        PID controller for the drone to reach the setpoint in the arena

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.pid()
        """

        actual_error = [0.0, 0.0, 0.0, 0.0]

        for i in range(4):
            actual_error[i] = self.drone_position[i] - self.setpoint[i]
            if i == 2:
                self.error[i] = (
                    self.throttle_filter.filter_throttle(self.drone_position[i])
                    - self.setpoint[i]
                )
            else:
                self.error[i] = self.drone_position[i] - self.setpoint[i]
            self.error_sum[i] = self.error_sum[i] + self.error[i]

            self.d_error[i] = self.error[i] - self.prev_error[i]
            self.prev_error[i] = self.error[i]

        self.pid_error.roll_error = actual_error[0]
        self.pid_error.pitch_error = actual_error[1]
        self.pid_error.throttle_error = actual_error[2]
        self.pid_error.yaw_error = actual_error[3]

        # Calculate PID outputs
        roll_output = (
            self.Kp[0] * self.error[0]
            + self.Ki[0] * self.error_sum[0]
            + self.Kd[0] * self.d_error[0]
        )

        pitch_output = (
            self.Kp[1] * self.error[1]
            + self.Ki[1] * self.error_sum[1]
            + self.Kd[1] * self.d_error[1]
        )
    
        throttle_output = (
            self.Kp[2] * self.error[2]
            + self.Ki[2] * self.error_sum[2]
            + self.Kd[2] * self.d_error[2]
        )
        
        yaw_output = (
            self.Kp[3] * self.error[3]
            + self.Ki[3] * self.error_sum[3]
            + self.Kd[3] * self.d_error[3]
        )

        # Update commands
        rc_roll = int(1500 - roll_output)
        rc_pitch = int(1500 + pitch_output)
        rc_throttle = int(1500 + throttle_output)
        rc_yaw = int(1500 + yaw_output)

        self.cmd.rc_roll = max(1000, min(2000, rc_roll))
        self.cmd.rc_pitch = max(1000, min(2000, rc_pitch))
        self.cmd.rc_throttle = max(1000, min(2000, rc_throttle))
        self.cmd.rc_yaw = max(1000, min(2000, rc_yaw))

        self.command_pub.publish(self.cmd)
        self.pid_error_pub.publish(self.pid_error)

    async def execute_callback(self, goal_handle):
        """
        Purpose:
        ---
        Callback to send feedback to client after receiving a waypoint, and also checks if the drone is stabilizing for three seconds

        Input Arguments:
        ---
        `goal_handle` :  [ < type of 1st input argument > ]
            The goal handle containing the request and feedback channels for the action client

        Returns:
        ---
        `result` :  [ NavToWaypoint.Result ]
            Result message to be sent to the action client

        Example call:
        ---
        await self.execute_callback(goal_handle)
        """

        if not self.is_stabilized:
            self.get_logger().info("Waiting for drone to stabilize at home position...")

            feedback_msg = NavToWaypoint.Feedback()

            while not self.is_stabilized:
                drone_in_home = self.is_drone_in_sphere(
                    self.drone_position,
                    type(
                        "obj",
                        (),
                        {
                            "request": type(
                                "obj",
                                (),
                                {
                                    "waypoint": type(
                                        "obj",
                                        (),
                                        {
                                            "position": type(
                                                "obj",
                                                (),
                                                {"x": 0.0, "y": 0.0, "z": 27.0},
                                            )
                                        },
                                    )
                                },
                            )
                        },
                    ),
                    0.6,
                )

                if drone_in_home:
                    if self.stabilization_start_time is None:
                        self.stabilization_start_time = self.dtime
                        self.get_logger().info(
                            "Drone in home position, starting stabilization timer"
                        )

                    stabilization_time = self.dtime - self.stabilization_start_time
                    self.get_logger().info(f"Stabilization time: {stabilization_time}")
                    if stabilization_time >= 3:
                        self.is_stabilized = True
                        self.get_logger().info("Drone stabilized at home position!")
                        break
                else:
                    self.stabilization_start_time = None

                # Publish feedback during stabilization
                feedback_msg.current_waypoint.pose.position.x = self.drone_position[0]
                feedback_msg.current_waypoint.pose.position.y = self.drone_position[1]
                feedback_msg.current_waypoint.pose.position.z = self.drone_position[2]
                goal_handle.publish_feedback(feedback_msg)

        self.setpoint[0] = goal_handle.request.waypoint.position.x
        self.setpoint[1] = goal_handle.request.waypoint.position.y
        self.setpoint[2] = goal_handle.request.waypoint.position.z
        self.get_logger().info(f"New Waypoint Set: {self.setpoint}")

        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0
        self.duration = self.dtime

        feedback_msg = NavToWaypoint.Feedback()

        while True:
            feedback_msg.current_waypoint.pose.position.x = self.drone_position[0]
            feedback_msg.current_waypoint.pose.position.y = self.drone_position[1]
            feedback_msg.current_waypoint.pose.position.z = self.drone_position[2]
            feedback_msg.current_waypoint.header.stamp.sec = self.max_time_inside_sphere

            goal_handle.publish_feedback(feedback_msg)

            drone_is_in_sphere = self.is_drone_in_sphere(
                self.drone_position, goal_handle, 0.6
            )

            if not drone_is_in_sphere and self.point_in_sphere_start_time is None:
                pass

            elif (
                drone_is_in_sphere
                and self.point_in_sphere_start_time is None
                and goal_handle.request.is_goal_point
            ):
                self.point_in_sphere_start_time = self.dtime
                self.get_logger().info("Drone in sphere for 1st time")

            elif (
                drone_is_in_sphere
                and self.point_in_sphere_start_time is not None
                and goal_handle.request.is_goal_point
            ):
                self.time_inside_sphere = self.dtime - self.point_in_sphere_start_time
                self.get_logger().info("Drone in sphere")
                self.get_logger().info(f"Time inside sphere: {self.time_inside_sphere}")

            elif (
                not drone_is_in_sphere
                and self.point_in_sphere_start_time is not None
                and goal_handle.request.is_goal_point
            ):
                self.get_logger().info("Drone out of sphere")
                self.point_in_sphere_start_time = None

            if self.time_inside_sphere > self.max_time_inside_sphere:
                self.max_time_inside_sphere = self.time_inside_sphere

            if goal_handle.request.is_goal_point and self.max_time_inside_sphere >= 3:
                print("Final goal reached!",goal_handle.request.is_goal_point)
                break
            if goal_handle.request.is_goal_point:
                print("Goal point reached, waiting for 3 seconds")
                
            elif not goal_handle.request.is_goal_point and self.is_drone_in_sphere(
                self.drone_position, goal_handle, 1.5
            ):
                print("Waypoint reached, ready for next point")
                break

        goal_handle.succeed()
        result = NavToWaypoint.Result()
        result.hov_time = self.dtime - self.duration
        return result

    def is_drone_in_sphere(self, drone_pos, sphere_center, radius):
        """
        Purpose:
        ---
        Checks if the drone is within a specified sphere centered at a given waypoint.

        Input Arguments:
        ---
        `drone_pos` :  [ list ]
            A list containing the x, y, and z coordinates of the drone's position.

        `sphere_center` :  [ object ]
            An object containing the waypoint information which includes the center coordinates of the sphere.

        `radius` :  [ float ]
            The radius of the sphere within which the drone's position is to be checked.

        Returns:
        ---
        `is_within_sphere` :  [ bool ]
            A boolean value indicating whether the drone is inside the sphere or not.


        Example call:
        ---
        self.is_drone_in_sphere([1.0, 2.0, 3.0], goal_handle, 1.5)
        """

        return (
            (drone_pos[0] - sphere_center.request.waypoint.position.x) ** 2
            + (drone_pos[1] - sphere_center.request.waypoint.position.y) ** 2
            + (drone_pos[2] - sphere_center.request.waypoint.position.z) ** 2
        ) <= radius**2


def main(args=None):
    """
    Purpose:
    ---
    Main function to initialize the node and run the Server

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

    waypoint_server = WayPointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        waypoint_server.get_logger().info("KeyboardInterrupt, shutting down.\n")
    finally:
        waypoint_server.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
