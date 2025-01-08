#!/usr/bin/env python3

'''
# Team ID:          1284
# Theme:            WareHouse Drone
# Author List:      Salo E S, Govind S Sarath, Arjun G Ravi, Devanand A
# Filename:         WD_1284_pico_controller.py
# Functions:        __init__, whycon_callback, altitude_set_pid, pitch_set_pid, roll_set_pid, pid, main, send_request, _signal_handler, shutdown, publish_filtered_data
# Global variables: MIN_ROLL, BASE_ROLL, MAX_ROLL, SUM_ERROR_ROLL_LIMIT, MIN_PITCH, BASE_PITCH, MAX_PITCH, SUM_ERROR_PITCH_LIMIT, MIN_THROTTLE, BASE_THROTTLE, MAX_THROTTLE, SUM_ERROR_THROTTLE_LIMIT, CMD
'''

# Importing the required libraries
import scipy.signal
import numpy as np
from rc_msgs.msg import RCMessage
from rc_msgs.srv import CommandBool
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PIDTune, PIDError
import rclpy
from rclpy.node import Node
import signal
from threading import Event

MIN_ROLL = 1200
BASE_ROLL = 1500
MAX_ROLL = 1700
SUM_ERROR_ROLL_LIMIT = 5000

MIN_PITCH = 1200
BASE_PITCH = 1500
MAX_PITCH = 1700
SUM_ERROR_PITCH_LIMIT = 5000

MIN_THROTTLE = 1250
BASE_THROTTLE = 1500
MAX_THROTTLE = 2000
SUM_ERROR_THROTTLE_LIMIT = 5000

CMD = [[], [], []]


class Swift_Pico(Node):
    def __init__(self):
        """
        
        purpose:
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
        Swift_Pico()
        
        """
        super().__init__('pico_controller')  # initializing ros node with name pico_controller
        
        # Add shutdown event to gracefully land the drone
        self._shutdown_event = Event()

        # Intializing the drone position
        self.drone_position = [0.0, 0.0, 0.0]

        # Setpoint [x,y,z]
        self.setpoint = [0, 0, 26]  
        
        # Drone command message
        self.cmd = RCMessage()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        self.Kp = [24, 26.5, 13] # .01
        self.Ki = [.115, .83, .050] # .001
        self.Kd = [425, 425, 152] # .1

        # [roll, pitch, throttle]
        # self.Kp = [0, 0, 15] # .01
        # self.Ki = [.0, .0, .001] # .001
        # self.Kd = [0, 0, 152] # .1

        # PID variables
        self.error = [0, 0, 0]
        self.prev_error = [0, 0, 0]
        self.sum_error = [0, 0, 0]
        self.change_in_error = [0, 0, 0]
        
        # Value limits
        self.max_values = [2000, 2000, 2000]  # [roll, pitch, throttle]
        self.min_values = [1000, 1000, 1000]  # [roll, pitch, throttle]

        # Error message
        self.pid_error = PIDError()

        self.sample_time = 0.060  # in seconds
        
        # Publishing /drone_command, /pid_error
        self.command_pub = self.create_publisher(RCMessage, '/drone/rc_command', 10)
        self.pid_error_pub = self.create_publisher(PIDError, '/pid_error', 10)

        # Subscribing to /whycon/poses, /throttle_pid, /pitch_pid, roll_pid
        self.create_subscription(PoseArray, "/whycon/poses", self.whycon_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)        

        #arm/disarm service client
        self.cli = self.create_client(CommandBool, "/drone/cmd/arming")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting again,,,,')
        self.req = CommandBool.Request()

        future = self.send_request() # ARMING THE DRONE
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.get_logger().info(response.data)

        self.timer = self.create_timer(self.sample_time, self.pid)
        
        # Register signal handlers to ensure a clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

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


    def whycon_callback(self, msg):
        """

        purpose:
        ---
        Callback function for whycon poses

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
        
        self.publish_filtered_data(roll = rc_roll,pitch = rc_pitch,throttle = raw_throttle)
        self.pid_error_pub.publish(self.pid_error)

    def _signal_handler(self, signum, frame):
        """
        
        purpose:
        ---
        Signal handler to initiate shutdown sequence
        
        Input Arguments:
        ---
        signum : int
            Signal number
        frame : frame
            Frame object
        
        Returns:
        ---
        None
        
        Example call:
        ---
        self._signal_handler(signum, frame)
        
        """
        self.get_logger().info(f'Received signal {signum}, initiating shutdown...')
        self._shutdown_event.set()

    def shutdown(self):
        """
        
        purpose:
        ---
        Safely land the drone by reducing throttle gradually
        
        Input Arguments:
        ---
        None
        
        Returns:
        ---
        bool
            True if successful, False otherwise
            
        Example call:
        ---
        self.shutdown()
        """
        try:
            # Maintain current roll and pitch, but slowly reduce throttle
            self.cmd.rc_roll = BASE_ROLL
            self.cmd.rc_pitch = BASE_PITCH
            self.cmd.rc_yaw = BASE_ROLL
            
            current_throttle = self.cmd.rc_throttle
            
            # Reduce throttle gradually (step of 10 every 0.1 seconds)
            while current_throttle > MIN_THROTTLE and not self._shutdown_event.is_set():
                current_throttle -= 10
                self.cmd.rc_throttle = current_throttle
                self.command_pub.publish(self.cmd)
                self.get_logger().info(f"Landing... Throttle: {current_throttle}")
                
                # Use asyncio.sleep instead of time.sleep to allow interruption
                try:
                    rclpy.sleep(0.1)  # Small delay between throttle reductions
                except Exception:
                    break
            
            # Final command to ensure minimum throttle
            self.cmd.rc_throttle = MIN_THROTTLE
            self.cmd.rc_roll = BASE_ROLL
            self.cmd.rc_pitch = BASE_PITCH
            self.cmd.rc_yaw = BASE_ROLL
            
            # Publish final command multiple times to ensure it gets through
            for _ in range(5):
                if self._shutdown_event.is_set():
                    break
                self.command_pub.publish(self.cmd)
                try:
                    rclpy.sleep(0.05)
                except Exception:
                    break
            
            self.get_logger().info("Landing complete")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error during shutdown: {str(e)}")
            return False


def main(args=None):
    """
    
    purpose:
    ---
    Main function to initialize the node and run the controller
    
    Input Arguments:
    ---
    args : list
        List of arguments
    
    Returns:
    ---
    None
    
    Example call:
    ---
    main()
    
    """
    rclpy.init(args=args)
    swift_pico = Swift_Pico()
    
    try:
        while rclpy.ok() and not swift_pico._shutdown_event.is_set():
            rclpy.spin_once(swift_pico, timeout_sec=0.1)
    except Exception as e:
        swift_pico.get_logger().error(f'Error during execution: {str(e)}')
    finally:
        # Always attempt to land safely
        if not swift_pico._shutdown_event.is_set():  # Only if not already shutting down
            swift_pico.get_logger().info('Initiating shutdown sequence...')
            swift_pico.shutdown()
        
        # Cleanup
        swift_pico.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()