import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    # Create all your nodes
    usb_cam_node = Node(
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
    )

    whycon_node = Node(
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
    )

    image_view_node = Node(
        package='image_view',
        executable='image_view',
        namespace='whycon_display',
        name='image_view',
        output='screen',
        remappings=[
            ('image', '/whycon/image_out')
        ]
    )

    crsf_node = Node(
        package='crsf_ros2',
        executable='crsf_ros',
        name='crsf_ros',
        output='screen'
    )

    # Wrap pico_controller in a TimerAction for delay
    pico_controller_delayed = TimerAction(
        period=5.0, 
        actions=[
            Node(
                package='swift_pico_hw',
                executable='pico_controller.py',
                name='pico_controller',
                output='screen'
            )
        ]
    )

    record_process = launch.actions.ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', 'task_4a', '/whycon/poses'],
        output='screen'
    )

    return LaunchDescription([
        usb_cam_node,
        whycon_node,
        image_view_node,
        crsf_node,
        pico_controller_delayed,  # This will start after 10 seconds
        record_process
    ])