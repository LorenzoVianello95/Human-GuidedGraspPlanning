# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

import numpy as np
import multiprocessing as mp
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf
import math
import argparse
import json
import os
import time

#lib for commander
import sys
sys.path.insert(0, '/home/lvianell/Desktop/Lorenzo_report')
from v6 import all_close, MoveGroupPythonIntefaceTutorial





def main():

    try:
        print "============ Press `Enter` to begin the tutorial by setting up the moveit_commander (press ctrl-d to exit) ..."
        raw_input()
        tutorial = MoveGroupPythonIntefaceTutorial()

        print "============ Press `Enter` to execute a movement using a pose goal ..."
        raw_input()
        tutorial.go_to_pose_goal()

        

    except rospy.ROSInterruptException:
      return
    except KeyboardInterrupt:
      return





if __name__ == '__main__':
    main()