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

import time

import multiprocessing as mp


subscribe_rgb= True
subscribe_depth=True
flag_is_at_home= True

# Instantiate CvBridge
bridge = CvBridge()

number_images=0
start_time = time.time()


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
        global number_images, start_time
        elapsed_time = time.time() - start_time
        flag= False
        if elapsed_time> 10:
            flag = True
        if flag:
            print("Rgb image written")
            cv2.imwrite("/home/lvianell/Desktop/Lorenzo_report/code_mask_NN/paletta_pictures/"+"color"+str(number_images)+'.png', cv2_img)
            flag=False;
            number_images+=1
            start_time= time.time()
        #else:
            #print("rgb discarted")
        #rospy.sleep(1)
        return

def image_callback_d(msg):
    #print("Received an image depth!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg,   "passthrough")
        rows,cols = cv2_img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-3,1)
        cv2_img = cv2.warpAffine(cv2_img,M,(cols,rows))
        cv2_img = np.array(cv2_img, dtype = np.float32)
        where_are_NaNs = np.isnan(cv2_img)
        cv2_img[where_are_NaNs] = 0.0
        #cv2_img[:,500:640]=0
        #cv2_img[:,0:100]=0
        #cv2_img[0:80,:]=0
        #cv2_img[400:480,:]=0
        #np.nan_to_num(cv2_img)
        cv2.normalize(cv2_img, cv2_img, 0, 1, cv2.NORM_MINMAX)
        #cv2.imshow("Image from my node", cv2_img)
        #cv2.waitKey(0)
    except CvBridgeError, e:
        print("bbb")
        print(e)
    else:
        cv2_img=np.expand_dims(cv2_img, axis=2)
        #print(cv2_img.tolist())
        # Save your OpenCV2 image as a jpeg 
        global subscribe_rgb, subscribe_depth
        if subscribe_depth: 
            print("depth image written")
            np.save("/home/lvianell/Desktop/Lorenzo_report/gqcnn/data/examples/single_object/"+"depth_0"+".npy", cv2_img)
            subscribe_depth=False
        #else:
            #print("discarted depth")
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
    #rospy.Subscriber(image_topic_d, Image, image_callback_d)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
