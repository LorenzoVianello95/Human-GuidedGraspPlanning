#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

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

#from policy import auxiliary_policy

#lib for commander
import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf
import math
import glob

import time, os, fnmatch, shutil
t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)

skip_firsts_frames = 0

subscribe_rgb= True
subscribe_depth=True
flag_is_at_home= True
pathDataset="/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/"

# Instantiate CvBridge
bridge = CvBridge()

def image_callback_rgb(msg):
    #print("Received an image rgb!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print("hhhh")
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        #time = msg.header.stamp
        global subscribe_rgb, skip_firsts_frames

        txtCounter = 0#len(glob.glob1(pathDataset,"*.txt"))
        if subscribe_rgb and skip_firsts_frames>10:
            print("Rgb image written")
            cv2.imwrite("/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/"+"color_"+str(txtCounter)+'.png', cv2_img)
            cv2.imwrite("/home/lvianell/Desktop/Lorenzo_report/datasets/color/"+timestamp+'.png', cv2_img)
            subscribe_rgb=False;
        else:
            skip_firsts_frames+=1
        #rospy.sleep(1)
        return

def image_callback_d(msg):
    #print("Received an image depth!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg,   "passthrough")
        rows,cols = cv2_img.shape
        #M = cv2.getRotationMatrix2D((cols/2,rows/2),-3,1)
        #cv2_img = cv2.warpAffine(cv2_img,M,(cols,rows))
        
        cv2_img = np.array(cv2_img, dtype = np.float32)
        min_depth= cv2_img.min()
        max_depth= cv2_img.max()
        distance_principal_point= cv2_img[rows/2,cols/2]
        where_are_NaNs = np.isnan(cv2_img)
        cv2_img[where_are_NaNs] = 0.0
        
        cv2.normalize(cv2_img, cv2_img, 0, 1, cv2.NORM_MINMAX)
        
        #cv2.imshow("Image from my node", cv_image_norm)
    except CvBridgeError, e:
        print("bbb")
        print(e)
    else:

        cv2_img=np.expand_dims(cv2_img, axis=2)
        #print(cv2_img.tolist())
        # Save your OpenCV2 image as a jpeg 
        global subscribe_rgb, subscribe_depth

        txtCounter =0# len(glob.glob1(pathDataset,"*.txt"))
        if subscribe_depth and skip_firsts_frames>10: 
            print("depth image written")
            np.save("/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/"+"depth_"+str(txtCounter)+".npy", cv2_img)
            np.save("/home/lvianell/Desktop/Lorenzo_report/datasets/depth/"+timestamp+".npy", cv2_img)
            
            text_file = open("/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/"+"camera_parameters_"+str(txtCounter)+".txt", "w")
            text_file.write("number_rows: "+str(rows)+"\n")
            text_file.write("number_cols: "+str(cols)+"\n")
            text_file.write("min_depth: "+str(min_depth)+"\n")
            text_file.write("max_depth: "+str(max_depth)+"\n")
            text_file.write("distance_principal_point: "+str(distance_principal_point)+"\n")
            
            #np.save("/home/lvianell/Desktop/gqcnn/data/examples/single_object/"+"depth_0"+".npy", cv2_img)
            #cv2.imwrite("/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/"+"depth_0"+".png", cv2_img)
            subscribe_depth=False
        elif skip_firsts_frames >10:
            print("ok images saved")
            print "_________________________________________________________"
            print " can I suggest you to take the mask from the color Image? "
            print " If yes run:"
            print " python3 build_mask.py"
            rospy.signal_shutdown("reason")
        #time = msg.header.stamp
        #cv2.imwrite("/home/lollo/Desktop/tesi/gqcnn/data/rgbd/data_2/"+"depth_0"+'.png', cv2_img*255)
        #auxiliary_policy()
        #subscribe_rgb= True
        #rospy.sleep(1)
        return

            
def main():
    print "============ Starting setup"
    rospy.init_node('image_listener')
    

    print "============ Receiving images from kinect"
    # Define your image topic
    image_topic_rgb = "/camera/rgb/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic_rgb, Image, image_callback_rgb)
    image_topic_d = "/camera/depth_registered/image"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic_d, Image, image_callback_d)
    # Spin until ctrl + c
    
    rospy.spin()

if __name__ == '__main__':
    main()
