import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    description_resource_parent = os.path.dirname(
        get_package_share_directory('openarm_description'))
    package_share = FindPackageShare('openarm_gazebo_hole_search')
    world = PathJoinSubstitution([package_share, 'worlds', 'hole_search.sdf'])
    xacro_file = PathJoinSubstitution(
        [package_share, 'urdf', 'openarm_right_gazebo.urdf.xacro'])
    controller_config = PathJoinSubstitution(
        [package_share, 'config', 'spiral_search.yaml'])
    rviz_config = PathJoinSubstitution(
        [package_share, 'rviz', 'hole_search.rviz'])

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]), value_type=str)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={'gz_args': ['-v 3 --render-engine ogre ', world]}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description, 'use_sim_time': True},
        ],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'openarm_right',
            '-topic', 'robot_description',
            '-allow_renaming', 'false',
        ],
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        ],
    )

    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '60',
            '--inactive',
        ],
    )

    effort_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'arm_effort_controller',
            '--controller-manager', '/controller_manager',
            '--controller-manager-timeout', '60',
            '--inactive',
        ],
    )

    controller = Node(
        package='openarm_gazebo_hole_search',
        executable='spiral_hole_search_controller',
        output='screen',
        parameters=[controller_config],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    unpause_world = ExecuteProcess(
        cmd=[
            'ign', 'service',
            '-s', '/world/hole_search/control',
            '--reqtype', 'ignition.msgs.WorldControl',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '3000',
            '--req', 'pause: false',
        ],
        output='screen',
    )

    activate_controllers = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'switch_controllers',
            '--activate', 'joint_state_broadcaster', 'arm_effort_controller',
            '--strict',
            '--controller-manager', '/controller_manager',
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Start RViz alongside Gazebo.'),
        SetEnvironmentVariable(
            'IGN_GAZEBO_RESOURCE_PATH',
            [
                description_resource_parent,
                ':',
                EnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', default_value=''),
            ],
        ),
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            [
                description_resource_parent,
                ':',
                EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value=''),
            ],
        ),
        gazebo,
        clock_bridge,
        robot_state_publisher,
        spawn_robot,
        TimerAction(period=3.0, actions=[joint_state_spawner]),
        TimerAction(period=4.0, actions=[effort_spawner]),
        TimerAction(period=5.0, actions=[controller]),
        TimerAction(period=5.0, actions=[rviz]),
        # A controller switch is completed from the Gazebo update loop.  Resume
        # the zero-gravity world first so Humble's controller manager can
        # process the switch request without waiting on a paused simulation.
        TimerAction(period=8.0, actions=[unpause_world]),
        TimerAction(period=9.0, actions=[activate_controllers]),
    ])
