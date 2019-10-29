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



def read_camera_parameters(file_path):
    file = open(file_path, "r")
    for line in file: 
        k,v =line.split()
        if k=="distance_principal_point:":
            return float(v)
    return 0

def extrasct_best_grasp(file_path):
    file = open(file_path, "r")
    for line in file: 
        line=line.replace("[","").replace("]","").replace("\n","")
        x, y, ang, depth, width, q, e =line.split()
        if e=="1":
            return float(x), float(y), float(depth), float(ang)
    return 0,0,0,0

def read_in_robot_frame(file_path):
    file = open(file_path, "r")
    coordinates=[]
    for line in file: 
        k, v =line.split()
        coordinates.append(float(v))
    return coordinates


def transform_in_robot_frame(x,y,yaw, pp_d ):
    # per trasformare lo yaw penso basti ruotarlo solo per lo yaw della trasformata o qualcosa del genere...

    #transformation in camera frame https://en.wikipedia.org/wiki/Pinhole_camera_model 
    focal_length= 525 # taken from http://wiki.ros.org/kinect_calibration/technical
    x*=pp_d/focal_length
    y*=pp_d/focal_length
    z=pp_d
    x/=1000 # desired in meters
    y/=1000
    z/=1000
    #transformation in robot frame TODO 

    return x,y,z,yaw


def main():
    print "============ Starting setup"

    camera_parameters_path= "/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/camera_parameters_0.txt"
    pp_distance= read_camera_parameters(camera_parameters_path)

    grasp_path= "/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/grasps_0.txt"
    grasp_x, grasp_y, grasp_z, grasp_yaw = extrasct_best_grasp(grasp_path)
    print grasp_x, grasp_y, grasp_z, grasp_yaw

    #  transform in robot frame
    #read grasps in robot frame
    robot_frame_path= "/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/grasp_in_robot_coordinates.yaml"

    z_rotation= grasp_yaw

    [x_position, y_position, z_position]= read_in_robot_frame(robot_frame_path)
    print x_position, y_position, z_position, z_rotation

    #TODO movegroup execute the path ... follow policy3

    #x_position=0; y_position=0.5; z_position=0.1; z_rotation=0

    try:
        print "============ Press `Enter` to begin the tutorial by setting up the moveit_commander (press ctrl-d to exit) ..."
        raw_input()
        tutorial = MoveGroupPythonIntefaceTutorial()

        print "============ Press `Enter` to execute a movement using a pose goal ..."
        raw_input()
        tutorial.go_to_pose_goal()

        print "============ Press `Enter` to set the end-effector joint ..."
        raw_input()
        #tutorial.command_gripper(0.035)

        #print "execute grasping to selected point\n"
        #print "============ Press `Enter` to execute a movement using a pose goal ..."
        #raw_input()
        #tutorial.pick(x=x_position, y=y_position, z=0.1)

        print "execute grasping to selected point\n"
        print "============ Press `Enter` to execute a movement using a pose goal ..."
        raw_input()
        tutorial.go_to_pose_goal(x=x_position, y=y_position, z=0.4, yaw= -z_rotation) #+0.04 -0.02

        print "============ Press `Enter` to set the end-effector joint ..."
        raw_input()
        tutorial.command_gripper(0.04)

        print "============ Press `Enter` to add a box to the planning scene ..."
        raw_input()
        tutorial.add_box(box_x= x_position, box_y=y_position, box_z=0.1, box_roll=0, box_pitch=0, box_yaw=-z_rotation)

        print "============ Press `Enter` to plan and display a Cartesian path ..."
        raw_input()
        cartesian_plan, fraction = tutorial.plan_cartesian_path(pick_place_escape=0, z_position=0.22)#z_position)#-0.01)

        print "============ Press `Enter` to execute a saved path ..."
        raw_input()
        tutorial.execute_plan(cartesian_plan)

        print "============ Press `Enter` to attach a Box to the Panda robot ..."
        raw_input()
        tutorial.attach_box()
        
        print "============ Press `Enter` to set the end-effector joint ..."
        raw_input()
        tutorial.command_gripper(0.020)

        print "============ Press `Enter` to plan and display a Cartesian path ..."
        raw_input()
        cartesian_plan, fraction = tutorial.plan_cartesian_path(pick_place_escape=1)

        print "============ Press `Enter` to execute a saved path ..."
        raw_input()
        tutorial.execute_plan(cartesian_plan)


        print "============ Press `Enter` to set the end-effector joint ..."
        raw_input()
        tutorial.command_gripper(0.035)

        print "============ Press `Enter` to detach the box from the Panda robot ..."
        raw_input()
        tutorial.detach_box()

        print "============ Press `Enter` to remove the box from the planning scene ..."
        raw_input()
        tutorial.remove_box()

    except rospy.ROSInterruptException:
      return
    except KeyboardInterrupt:
      return





if __name__ == '__main__':
    main()