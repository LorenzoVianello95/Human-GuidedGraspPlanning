from __future__ import print_function

import numpy as np
import yaml
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from message_filters import Subscriber
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import pcl
import matplotlib.pyplot as plt
import thread
from math import pi
from numpy import *
import math
import sys
from tf.transformations import quaternion_from_euler

import pcl
from sensor_msgs import point_cloud2

import glob, os

def extrasct_best_grasp(file_path):
    file = open(file_path, "r")
    for line in file:
        line=line.replace("[","").replace("]","").replace("\n","")
        print(line.split())
        x, y, ang, depth, width, q, e =line.split()
        if e=="1":
            return float(x), float(y)
    return 0,0,

class mainTransformToRobotCoordinates:
    def __init__(self, debug=False, init_ros_node=False):
        self.debug = debug
        self.menu_str = "\nChoose an action:\n 1 - Compute transformation\n 2 - Exit\n\n>"
        self.frame_to_skip = 10
        self.cloud_points = []
        self.objpoints = []
        self.cv2_img = None
        self.depth_img = None

        self.buisy = False

        self.bridge = CvBridge()

        if init_ros_node:
            rospy.init_node('main_transformation', anonymous=True)

        self.image_sub = Subscriber("/camera/rgb/image_rect_color", Image)
        self.depth_sub = Subscriber("/camera/depth_registered/points", PointCloud2)
        self.tss = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub],
                                                               queue_size=1, slop=0.5)
        #time.sleep(0.5)
        self.tss.registerCallback(self.callback_decide_everything)

    def get_int(self, prompt):
        while True:
            user_answer = raw_input(prompt)
            print(user_answer)
            if (user_answer.isdigit()):
                return int(user_answer)
            else:
                print("\nInvalid input please enter a number!\n")

    def decide_everything(self, threadName, delay):
        action = self.get_int(self.menu_str)
        if action == 0:
            print("CTRL + C to exit")

        # test addition
        elif action == 1:
            #detect and save position in image and of robot
            self.transform_in_robot_frame()

        self.buisy = False

    def get_frames(self):
        self.cv2_img = None
        self.depth_img = None
        while self.cv2_img is None:
            try:
                self.cv2_img = self.bridge.imgmsg_to_cv2(self.rgb_raw, "bgr8")
            except CvBridgeError as e:
                print(e)

    def callback_decide_everything(self, img, depth):

        if (self.frame_to_skip > 0):
            self.frame_to_skip = self.frame_to_skip - 1
        else:
            self.rgb_raw = img
            self.depth_raw = depth

            # print("data updated")
            if (not self.buisy):
                self.buisy = True
                # Create two threads as follows
                try:
                    thread.start_new_thread(self.decide_everything, ("Thread-1", 2,))
                except:
                    print("Error: unable to start thread")

    def transform_in_robot_frame(self):

        # number of points
        number_of_points = 1

        #read grasps_0.txt
        #grasp_path= "../datasets/grasp/grasps_0.txt"
        list_of_files = glob.glob('/home/lvianell/Desktop/Lorenzo_report/datasets/grasp/*')
        grasp_path= max(list_of_files, key=os.path.getctime)
        grasp_x, grasp_y = extrasct_best_grasp(grasp_path)
        print (grasp_x, grasp_y)


        cloud_points = list(point_cloud2.read_points(self.depth_raw, skip_nans=False, field_names=("x", "y", "z")))
        index = (int(grasp_y) * 640 + int(grasp_x))

        imagePoints =[cloud_points[index]]
        print("In depth camera")
        print(imagePoints)
        A = np.zeros((number_of_points, 3), np.float32)

        for i in range(number_of_points):
            A[i, 0] = imagePoints[i][0]
            A[i, 1] = imagePoints[i][1]
            A[i, 2] = imagePoints[i][2]

        print ("Points A")
        print (A)
        print ("")

        stream = open("../datasets/dexnet_maskcnn_human/extrinsicGuess_rvecs.yaml", "r")
        rvecs = yaml.load(stream)

        stream = open("../datasets/dexnet_maskcnn_human/extrinsicGuess_tvecs.yaml", "r")
        tvecs = yaml.load(stream)

        rvecs = np.array(rvecs)
        #rvecs, _ = cv2.Rodrigues(rvecs)

        tvecs = np.array(tvecs)

        print("loaded rvecs: " + str(rvecs))
        print("loaded tvecs: " + str(tvecs))

        ret_R = rvecs

        ret_t = tvecs

        A2 = np.dot(ret_R , transpose(A)) + tile(ret_t, (1, number_of_points))
        A2 = transpose(A2)



        print ("Points A2 = rvec * A + tvecs")
        print (A2)
        print ("")

        stream = open("../datasets/dexnet_maskcnn_human/grasp_in_robot_coordinates.yaml", "w")
        yaml.dump(A2[0].tolist(), stream)

        return A2

def main(args):
    print("============ Starting setup")

    cf = mainTransformToRobotCoordinates(debug=False, init_ros_node=True)
    # rospy.init_node('camera_feed', anonymous=True )
    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
