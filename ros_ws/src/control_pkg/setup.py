from setuptools import setup
from glob import glob

package_name = 'control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mario',
    maintainer_email='mario.hurtado@udem.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'closest_object = control_pkg.closest_object:main',
            'color_detection = control_pkg.color_detection_node:main',
            'lidar_avoid = control_pkg.lidar_avoid:main',
            'cmd_mux = control_pkg.cmd_mux:main',
            'color_tracking = control_pkg.cv_object_tracking_color:main'
        ],
    },
)

