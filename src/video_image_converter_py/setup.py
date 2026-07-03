from glob import glob
import os

from setuptools import setup


package_name = 'video_image_converter_py'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (
            os.path.join('share', package_name),
            [package_name + '/video_image_converter_params.yaml'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dotX Automation s.r.l.',
    maintainer_email='info@dotxautomation.com',
    description='Generic ROS 2 video file to Image topic converter.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'video_image_converter = video_image_converter_py.video_image_converter_app:main',
        ],
    },
)
