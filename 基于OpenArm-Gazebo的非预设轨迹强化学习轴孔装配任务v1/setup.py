from glob import glob
from setuptools import find_packages, setup

package_name = 'openarm_gazebo_hole_search'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenArm Developer',
    maintainer_email='openarm@example.com',
    description='Gazebo simulation for OpenArm YZ spiral hole search.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'spiral_hole_search_controller = '
            'openarm_gazebo_hole_search.spiral_hole_search_controller:main',
            'rl_hole_search_controller = '
            'openarm_gazebo_hole_search.rl_hole_search_controller:main',
            'sac_runtime_controller = '
            'openarm_gazebo_hole_search.sac_runtime_controller:main',
        ],
    },
)
