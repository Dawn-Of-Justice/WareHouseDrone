import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, GroupAction

def generate_launch_description():
   # Define the nodes that need delay
   pico_server_node = Node(
       package='swift_pico_hw',
       executable='pico_server_5.py',
       name='pico_server',
    #    output='screen'
   )

   pico_client_node = Node(
       package='swift_pico_hw',
       executable='pico_client_5.py',
       name='pico_client',
       output='screen'
   )

   waypoint_service_node = Node(
       package='swift_pico_hw',
       executable='path_planning_service.py',
       name='path_planning_service',
       output='screen'
   )

   # Wrap delayed nodes with TimerAction
   delayed_pico_server = TimerAction(
       period=8.0,
       actions=[pico_server_node]
   )

   delayed_pico_client = TimerAction(
       period=4.0,
       actions=[pico_client_node]
   )

   delayed_waypoint_service = TimerAction(
       period=2.0,
       actions=[waypoint_service_node]
   )

   # Group nodes for coordinated shutdown
   node_group = GroupAction([
       Node(
           package='usb_cam',
           name='usb_cam',
           executable='usb_cam_node_exe',
           output='screen',
           parameters=[{
               'video_device': '/dev/video2',
               'image_width': 1920,
               'image_height': 1080,
               'pixel_format': 'mjpeg2rgb',
               'io_method': 'mmap',
               'framerate': 30.0,
               'camera_frame_id': 'usb_cam',
               'av_device_format': 'YUV422P',
           }]
       ),

       Node(
           package='whycon',
           name='whycon',
           namespace='whycon',
           executable='whycon',
           output='screen',
           parameters=[{
               'targets': 1,
               'name': 'whycon',
               'outer_diameter': 0.38,
               'inner_diameter': 0.14,
           }]
       ),

       Node(
           package='image_view',
           executable='image_view',
           namespace='whycon_display',
           name='image_view',
           output='screen',
           remappings=[
               ('image', '/whycon/image_out')
           ]
       ),

       Node(
           package='crsf_ros2',
           executable='crsf_ros',
           name='crsf_ros',
       ),

       delayed_waypoint_service,
       delayed_pico_client,
       delayed_pico_server
   ])

   return LaunchDescription([
       node_group
   ])