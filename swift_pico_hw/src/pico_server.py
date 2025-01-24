#!/usr/bin/env python3
"""
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
# Filename:         WD_1284_pico_server.py
# Functions:        __init__, whycon_callback, altitude_set_pid, roll_set_pid, pitch_set_pid, yaw_set_pid, odometry_callback, publish_filtered_data, pid, execute_callback, is_drone_in_sphere, shutdown, main
# Global variables: MIN_ROLL, BASE_ROLL, MAX_ROLL, SUM_ERROR_ROLL_LIMIT, MIN_PITCH, BASE_PITCH, MAX_PITCH, SUM_ERROR_PITCH_LIMIT, MIN_THROTTLE, BASE_THROTTLE, MAX_THROTTLE, SUM_ERROR_THROTTLE_LIMIT, CMD
"""


import math
import scipy
import scipy.signal
from tf_transformations import euler_from_quaternion
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from waypoint_navigation.action import NavToWaypoint
from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PIDTune, PIDError
from nav_msgs.msg import Odometry
import numpy as np

MIN_ROLL = 1200
BASE_ROLL = 1460
MAX_ROLL = 1700
SUM_ERROR_ROLL_LIMIT = 5000

MIN_PITCH = 1200
BASE_PITCH = 1460
MAX_PITCH = 1700
SUM_ERROR_PITCH_LIMIT = 5000

MIN_THROTTLE = 1250
BASE_THROTTLE = 1460
MAX_THROTTLE = 2000
SUM_ERROR_THROTTLE_LIMIT = 5000

CMD = [[], [], []]

class WayPointServer(Node):

    def __init__(self):
        """
        purpose:
        ---
        Initialize the WayPoint Server Node with ROS2 subscribers, publishers, and action server

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
        super().__init__("waypoint_server") # Initialize the node
        
        self.pid_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()

        self.time_inside_sphere = 0 # Variable to store Time spent inside the sphere
        self.max_time_inside_sphere = 0 # Variable to store Maximum time spent inside the sphere
        self.point_in_sphere_start_time = None # Variable to store the time when the drone enters the sphere
        self.duration = 0 # Variable to store the time when the drone enters the sphere

        self.drone_position = [0.0, 0.0, 0.0, 0.0] # [x, y, z, yaw]
        self.setpoint = [0, 0, 26, 0] # Target position [x, y, z, yaw]
        self.dtime = 0

        self.sample_time = 0.06 # Sample time

        # Initialize PID controller parameters
        self.cmd = RCMessage()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # self.Kp = [0, 0, 14] # .01
        # self.Ki = [0, 0, 0.060] # .001
        # self.Kd = [0, 0, 135.5] # .1

        self.Kp = [23.6, 23.6, 14] # .01
        self.Ki = [.115, .115, .060] # .001
        self.Kd = [420, 420, 135.5] # .1

        # PID controller variables
        self.error = [0.0, 0.0, 0.0, 0.0]  # Current errors
        self.prev_error = [0.0, 0.0, 0.0, 0.0]  # Previous errors
        self.sum_error = [-1000, 0, 0.0, 0.0]  # Sum of errors (for integral)
        self.change_in_error = [0.0, 0.0, 0.0, 0.0]  # Change in error (for derivative)

        # Value limits
        self.max_values = [2000, 2000, 2000]  # [roll, pitch, throttle]
        self.min_values = [1000, 1000, 1000]  # [roll, pitch, throttle]

        self.pid_error = PIDError()
        
        # Publishers and Subscribers
        self.command_pub = self.create_publisher(RCMessage, '/drone/rc_command', 10)
        self.pid_error_pub = self.create_publisher(PIDError, '/pid_error', 10)

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

        # Service Client for arming and disarming the drone
        self.cli = self.create_client(CommandBool, "/drone/cmd/arming")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting again,,,,')
        self.req = CommandBool.Request()

        future = self.send_request() # ARMING THE DRONE
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info(response.data)

        self.timer = self.create_timer(self.sample_time, self.pid, callback_group=self.pid_callback_group)
    
    def send_request(self):
        """
        
        purpose:
        ---
        Send request to arm the drone
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        Future object
            Future object representing the service call
        
        Example call:
        ---
        send_request()
        
        """
        self.req.value = True
        return self.cli.call_async(self.req)

    def whycon_callback(self, msg):
        """
        purpose:
        ---
        Update drone position based on WhyCon camera pose data

        Input Arguments:
        ---
        msg : PoseArray
            ROS message containing drone position from WhyCon camera

        Returns:
        ---
        None

        Example call:
        ---
        whycon_callback(pose_array_message)
        """
        self.drone_position[0] = msg.poses[0].position.x
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z
        self.dtime = msg.header.stamp.sec

    def altitude_set_pid(self, alt):
        """
        purpose:
        ---
        Set PID proportional, integral, and derivative gains for altitude control

        Input Arguments:
        ---
        alt : PIDTune
            ROS message containing PID tuning parameters for altitude

        Returns:
        ---
        None

        Example call:
        ---
        altitude_set_pid(pid_tune_message)
        """
        self.Kp[2] = alt.kp * 0.01
        self.Ki[2] = alt.ki * 0.001
        self.Kd[2] = alt.kd * 0.1

    def roll_set_pid(self, msg):
        """
        purpose:
        ---
        Set PID proportional, integral, and derivative gains for roll control

        Input Arguments:
        ---
        msg : PIDTune
            ROS message containing PID tuning parameters for roll

        Returns:
        ---
        None

        Example call:
        ---
        roll_set_pid(pid_tune_message)
        """
        self.Kp[0] = msg.kp * 0.01
        self.Ki[0] = msg.ki * 0.001
        self.Kd[0] = msg.kd * 0.1

    def pitch_set_pid(self, msg):
        """
        purpose:
        ---
        Set PID proportional, integral, and derivative gains for pitch control

        Input Arguments:
        ---
        msg : PIDTune
            ROS message containing PID tuning parameters for pitch

        Returns:
        ---
        None

        Example call:
        ---
        pitch_set_pid(pid_tune_message)
        """
        self.Kp[1] = msg.kp * 0.01
        self.Ki[1] = msg.ki * 0.001
        self.Kd[1] = msg.kd * 0.1

    def yaw_set_pid(self, msg):
        """
        purpose:
        ---
        Set PID proportional, integral, and derivative gains for yaw control

        Input Arguments:
        ---
        msg : PIDTune
            ROS message containing PID tuning parameters for yaw

        Returns:
        ---
        None

        Example call:
        ---
        yaw_set_pid(pid_tune_message)
        """
        self.Kp[3] = msg.kp * 0.01
        self.Ki[3] = msg.ki * 0.001
        self.Kd[3] = msg.kd * 0.1

    def odometry_callback(self, msg):
        """
        purpose:
        ---
        Process odometry message to extract and convert drone orientation 
        from quaternion to Euler angles

        Input Arguments:
        ---
        msg : Odometry
            ROS message containing drone orientation in quaternion format

        Returns:
        ---
        None

        Example call:
        ---
        odometry_callback(odometry_message)
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

    def publish_filtered_data(self, roll, pitch, throttle):
        """
        
        purpose:
        ---
        Publish the filtered data to the drone
        
        Input Arguments:
        ---
        roll : int
            The roll value
        pitch : int
            The pitch value
        throttle : int
            The throttle value
            
        Returns:
        ---
        None
        
        Example call:
        ---
        self.publish_filtered_data(roll, pitch, throttle)
        
        """

        self.cmd.rc_throttle = int(throttle)
        self.cmd.rc_roll = int(roll)
        self.cmd.rc_pitch = int(pitch)
        self.cmd.rc_yaw = int(1500)

        # BUTTERWORTH FILTER low pass filter
        span = 15
        for index, val in enumerate([roll, pitch, throttle]):
            CMD[index].append(val)
            if len(CMD[index]) == span:
                CMD[index].pop(0)
            if len(CMD[index]) != span-1:
                return
            order = 3 # determining order 
            fs = 30 # to keep in order same as hz topic runs
            fc = 4 
            nyq = 0.5 * fs
            wc = fc / nyq
            b, a = scipy.signal.butter(N=order, Wn=wc, btype='lowpass', analog=False, output='ba')
            filtered_signal = scipy.signal.lfilter(b, a, CMD[index])
            
            if index == 0:
                rc_roll = int(filtered_signal[-1])
                if rc_roll > MAX_ROLL:
                    self.cmd.rc_roll = MAX_ROLL
                elif rc_roll < MIN_ROLL:
                    self.cmd.rc_roll = MIN_ROLL
                else:
                    self.cmd.rc_roll = rc_roll
            elif index == 1:
                rc_pitch = int(filtered_signal[-1])
                if rc_pitch > MAX_PITCH:
                    self.cmd.rc_pitch = MAX_PITCH
                elif rc_pitch < MIN_PITCH:
                    self.cmd.rc_pitch = MIN_PITCH
                else:
                    self.cmd.rc_pitch = rc_pitch
            elif index == 2:
                rc_throttle = int(filtered_signal[-1])
                if rc_throttle > MAX_THROTTLE:
                    self.cmd.rc_throttle = MAX_THROTTLE
                elif rc_throttle < MIN_THROTTLE:
                    self.cmd.rc_throttle = MIN_THROTTLE
                else:
                    self.cmd.rc_throttle = rc_throttle


        self.command_pub.publish(self.cmd)


    def pid(self):
        """
        purpose:
        ---
        Implement PID control to stabilize the drone in the air.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        pid()
        """
        for i in range(3):
            self.error[i] = self.drone_position[i] - self.setpoint[i]
            self.change_in_error[i] = self.error[i] - self.prev_error[i]
            
            # Add anti-windup
            if i == 0:  # Roll
                self.sum_error[i] = np.clip(self.sum_error[i] + self.error[i], -SUM_ERROR_ROLL_LIMIT, SUM_ERROR_ROLL_LIMIT)
            elif i == 1:  # Pitch
                self.sum_error[i] = np.clip(self.sum_error[i] + self.error[i], -SUM_ERROR_PITCH_LIMIT, SUM_ERROR_PITCH_LIMIT)
            else:  # Throttle
                self.sum_error[i] = np.clip(self.sum_error[i] + self.error[i], -SUM_ERROR_THROTTLE_LIMIT, SUM_ERROR_THROTTLE_LIMIT)
                
            self.prev_error[i] = self.error[i]
            
        self.pid_error.roll_error = self.error[0]
        self.pid_error.pitch_error = self.error[1]
        self.pid_error.throttle_error = self.error[2]
        
        # Calculate PID outputs
        roll_output = (
            self.Kp[0] * self.error[0]
            + self.sum_error[0] * self.Ki[0]
            + self.Kd[0] * self.change_in_error[0]
        )
        pitch_output = (
            self.Kp[1] * self.error[1]
            + self.sum_error[1] * self.Ki[1]
            + self.Kd[1] * self.change_in_error[1]
        )
        throttle_output = (
            self.Kp[2] * self.error[2]
            + self.sum_error[2] * self.Ki[2]
            + self.Kd[2] * self.change_in_error[2]
        )

        # Update commands
        rc_roll = int(BASE_ROLL - roll_output)
        rc_pitch = int(BASE_PITCH + pitch_output) 
        raw_throttle = int(BASE_THROTTLE + throttle_output)     
        print(raw_throttle)
        self.publish_filtered_data(roll = rc_roll,pitch = rc_pitch,throttle = raw_throttle)
        self.pid_error_pub.publish(self.pid_error)



    async def execute_callback(self, goal_handle):
        """
        purpose:
        ---
        Execute waypoint navigation action, tracking drone's progress 
        towards the specified waypoint

        Input Arguments:
        ---
        goal_handle : ActionGoalHandle
            ROS action goal handle containing waypoint navigation request

        Returns:
        ---
        NavToWaypoint.Result
            Result of waypoint navigation action

        Example call:
        ---
        execute_callback(goal_handle)
        """
        self.get_logger().info("Executing goal...")
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
            # Update feedback message with current waypoint
            feedback_msg.current_waypoint.pose.position.x = self.drone_position[0]
            feedback_msg.current_waypoint.pose.position.y = self.drone_position[1]
            feedback_msg.current_waypoint.pose.position.z = self.drone_position[2]
            feedback_msg.current_waypoint.header.stamp.sec = self.max_time_inside_sphere

            goal_handle.publish_feedback(feedback_msg)

            drone_is_in_sphere = self.is_drone_in_sphere(
                self.drone_position, goal_handle, 0.8
            )  

            if not drone_is_in_sphere and self.point_in_sphere_start_time is None:
                pass
            elif drone_is_in_sphere and self.point_in_sphere_start_time is None:
                self.point_in_sphere_start_time = self.dtime
                self.get_logger().info("Drone in sphere for 1st time")
            elif drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                self.time_inside_sphere = self.dtime - self.point_in_sphere_start_time
                self.get_logger().info("Drone in sphere")
                self.get_logger().info(f"Time inside sphere: {self.time_inside_sphere}")
            elif not drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                self.point_in_sphere_start_time = None

            if self.time_inside_sphere > self.max_time_inside_sphere:
                self.max_time_inside_sphere = self.time_inside_sphere

            if goal_handle.request.is_goal_point:
                if self.max_time_inside_sphere >= 3:
                    break
            else:
                # For intermediate points, just need to reach the sphere once
                if self.is_drone_in_sphere(self.drone_position, goal_handle, 0.8):
                    break
        
        # Send success message to client
        goal_handle.succeed()
        result = NavToWaypoint.Result()
        result.hov_time = self.dtime - self.duration
        return result

    def is_drone_in_sphere(self, drone_pos, sphere_center, radius):
        """
        purpose:
        ---
        Check if the drone is within a specified radius of a target point

        Input Arguments:
        ---
        drone_pos : list
            Current drone position [x, y, z, yaw]
        sphere_center : object
            Goal handle containing the target waypoint
        radius : float
            Radius of the acceptance sphere around the waypoint

        Returns:
        ---
        bool
            True if drone is inside the sphere, False otherwise

        Example call:
        ---
        is_drone_in_sphere(current_position, goal_handle, 0.4)
        """
        return (
            (drone_pos[0] - sphere_center.request.waypoint.position.x) ** 2
            + (drone_pos[1] - sphere_center.request.waypoint.position.y) ** 2
            + (drone_pos[2] - sphere_center.request.waypoint.position.z) ** 2
        ) <= radius**2

        
def main(args=None):
    """
    purpose:
    ---
    Initialize ROS2 node, create WayPointServer, and spin the MultiThreadedExecutor

    Input Arguments:
    ---
    args : list, optional
        Command line arguments for ROS2 initialization

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

if __name__ == '__main__':
    main()
