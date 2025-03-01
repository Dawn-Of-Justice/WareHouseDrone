#!/usr/bin/env python3

"""
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
# Filename:         WD_1284_pico_server_5.py
# Functions:        __init__, butter_lowpass, filter_throttle, disarm, arm, whycon_callback, altitude_set_pid, roll_set_pid, pitch_set_pid, yaw_set_pid, pid, execute_callback, is_drone_in_sphere, main
# Global variables: None
"""

from functools import partial
import time
import rclpy
import rclpy
import numpy as np
import scipy.signal
from rclpy.node import Node
from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from rclpy.action import ActionServer
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PIDTune, PIDError
from rclpy.executors import MultiThreadedExecutor
from waypoint_navigation.action import NavToWaypoint
from rclpy.callback_groups import ReentrantCallbackGroup

MIN_ROLL = 1200
BASE_ROLL = 1500
MAX_ROLL = 1700
SUM_ERROR_ROLL_LIMIT = 2000

MIN_PITCH = 1200
BASE_PITCH = 1500
MAX_PITCH = 1700
SUM_ERROR_PITCH_LIMIT = 2000

MIN_THROTTLE = 1250
BASE_THROTTLE = 1500
MAX_THROTTLE = 2000
SUM_ERROR_THROTTLE_LIMIT = 2000

CMD = [[], [], []]


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

        # Sphere tracking variables
        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.duration = 0
        self.dtime = 0
        
        # Intializing the drone position
        self.drone_position = [0.0, 0.0, 0.0, 0.0]

        self.pid_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        
        self.is_stabilized = False
        
        # Setpoint [x,y,z]
        self.setpoint = [0, 0, 26]  

        self.cmd = RCMessage()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # [roll, pitch, throttle]
        self.Kp = [25.54, 26.15, 13.98] # .01
        self.Ki = [.057, .069, .058] # .001
        self.Kd = [403, 402.5, 147.9] # .1
        

        self.error = [0, 0, 0]
        self.prev_error = [0, 0, 0]
        self.sum_error = [-1000, 0, 0]
        self.change_in_error = [0, 0, 0]
        
        # Value limits
        self.max_values = [2000, 2000, 2000]  # [roll, pitch, throttle]
        self.min_values = [1000, 1000, 1000]  # [roll, pitch, throttle]
        
        # Error message
        self.pid_error = PIDError()
        
        self.sample_time = 0.04 # in seconds

        self.command_pub = self.create_publisher(RCMessage, '/drone/rc_command', 10)
        self.pid_error_pub = self.create_publisher(PIDError, "/pid_error", 10)

        self.create_subscription(PoseArray, "/whycon/poses", self.whycon_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)
        self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)

        #arm/disarm service client
        self.cli = self.create_client(CommandBool, "/drone/cmd/arming")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting again,,,,')
        self.req = CommandBool.Request()
        
        future = self.send_request() # ARMING THE DRONE
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info(response.data)
        
        # Action Server
        self._action_server = ActionServer(
            self,
            NavToWaypoint,
            "waypoint_navigation",
            self.execute_callback,
            callback_group=self.action_callback_group,
        )

        self.timer = self.create_timer(
            self.sample_time, self.pid, callback_group=self.pid_callback_group
        )
        
        
        self.landing_requested = False
        
    def whycon_callback(self, msg):
        """
        purpose:
        ---
        Callback function for whycon poses and update sphere tracking

        Input Arguments:
        ---
        msg : PoseArray
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
        purpose:
        ---
        Set the PID values for altitude

        Input Arguments:
        ---
        alt : PIDTune
            The message containing the PID values

        Returns:
        ---
        None

        Example call:
        ---
        self.altitude_set_pid(alt)
        """
        self.Kp[2] = alt.kp * 0.01
        self.Ki[2] = alt.ki * 0.001
        self.Kd[2] = alt.kd * 0.1

    def pitch_set_pid(self, pitch):
        """
        purpose:
        ---
        Set the PID values for pitch

        Input Arguments:
        ---
        pitch : PIDTune
            The message containing the PID values

        Returns:
        ---
        None

        Example call:
        ---
        self.pitch_set_pid(pitch)
        """
        self.Kp[1] = pitch.kp * 0.01
        self.Ki[1] = pitch.ki * 0.001
        self.Kd[1] = pitch.kd * 0.1

    def roll_set_pid(self, roll):
        """
        purpose:
        ---
        Set the PID values for roll

        Input Arguments:
        ---
        roll : PIDTune
            The message containing the PID values

        Returns:
        ---
        None

        Example call:
        ---
        self.roll_set_pid(roll)
        """
        self.Kp[0] = roll.kp * 0.01
        self.Ki[0] = roll.ki * 0.001
        self.Kd[0] = roll.kd * 0.1
    
    def send_request(self):
        """
        purpose:
        ---
        Send request to arm the drone using the service client
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        future : Future
            Future object
            
        Example call:
        ---
        self.send_request()
        """
        self.req.value = True
        return self.cli.call_async(self.req)
    
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
        
        self.publish_filtered_data(roll = rc_roll, pitch = rc_pitch, throttle = raw_throttle)
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
        
        # Store old setpoint
        old_setpoint = self.setpoint.copy()
        
        # Update to new setpoint
        self.setpoint[0] = goal_handle.request.waypoint.position.x
        self.setpoint[1] = goal_handle.request.waypoint.position.y
        self.setpoint[2] = goal_handle.request.waypoint.position.z
        
        # Reset integral component when setpoint changes significantly
        if (abs(old_setpoint[0] - self.setpoint[0]) > 3 or 
            abs(old_setpoint[1] - self.setpoint[1]) > 3 or
            abs(old_setpoint[2] - self.setpoint[2]) > 3):
            self.sum_error = [0, 0, 0]
            
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
                self.drone_position, goal_handle, 0.8
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
                self.get_logger().info(f"Target setpoint: {self.setpoint}")

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
                self.get_logger().info("Goal reached! Initiating request for next point...")
                if goal_handle.request.avada_kedavra == True:
                    self.get_logger().info("Initiating landing sequence...")
                    await self.land_safely()

                break
            if goal_handle.request.is_goal_point:
                self.get_logger().info("Goal point reached, waiting for 3 seconds")
                pass
                
            elif not goal_handle.request.is_goal_point and self.is_drone_in_sphere(
                self.drone_position, goal_handle, 1.8
            ):
                self.get_logger().info("Waypoint reached, ready for next point")
                break
            

        goal_handle.succeed()
        result = NavToWaypoint.Result()
        result.hov_time = float(self.dtime - self.duration)
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
            
    def waypoint_callback(self, request, response):
        if hasattr(request, 'land_drone') and request.land_drone:
            self.landing_requested = True
            response.landing_complete = True
            return response
    
    async def land_safely(self):
        """
        Purpose:
        ---
        Safely land the drone by disarming it after controlled descent.
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        None
        
        Example call:
        ---
        await self.land_safely()
        
        """
        self.get_logger().info("Beginning controlled descent...")
        
        initial_height = self.drone_position[2]
        descent_rate = 0.5 
        duration = 0.1  # 100ms between iterations
        
        current_z = initial_height
        while current_z < 30.5:  # Disarm request will be sent when drone reaches the point 30.5
            current_z += descent_rate * duration
            self.setpoint[2] = current_z
            
            # Create an awaitable sleep using ROS 2's executor
            future = rclpy.task.Future()
            self.executor.create_task(partial(future.set_result, None))
            time.sleep(duration)  # This will allow other callbacks to run
            await future
            
        # Final disarm
        self.get_logger().info("Landing complete, disarming...")
        self.req.value = False
        await self.cli.call_async(self.req)

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