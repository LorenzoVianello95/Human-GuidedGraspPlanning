In this folder there are all file necessary for pick and place on real robot:

- ```python put_robot_initial_conf.py``` this is used to set the robot in a way that it doesn't cover the camera during the data reading.

- ```python save_depth_color.py``` this is used to read the color and the depth image from the RGBD sensor, these are saved in Lorenzo_report/data/human_data and can be used in the future to do the learning.

- ```python3 build_mask.py``` this is used to build the binary mask using the RGB image, also the mask is saved in the same directory, it's important to notice that in this case we must use python3.

- ```python dexnet_grasps_extract.py``` this is use the depth image, the mask and the gqcnn folder to extract the best grasps, during the execution will be asked if you prefere to use or not the mask and if you want to infer in the best grasp selection, all the results are then saved in the same folder.

- ```python add_grasp_position_in_robot_coordinates.py``` this is used to transform the grasps in camera frame in robot frame.

- ```python execute_grasp.py``` this is used to execute the grasp precedently calculated, it use the v6.py file in Lorenzo_Report, the grasp is composed by the following actions:
	- if the robot is not in Home position go there.
	- go over the grasp (around 40 cm).
	- open the gripper (ATT: this passege many times gives errors).
	- go down executing a cartesian path (ATT: also this passage if the grasp is to far returns errors, probably given by to much load is some joints).
	- close gripper (ATT: this passege many times gives errors).
	- TODO: check if the grasp was successfull or not and save that information.
	- go up executing cartesian path.
	- open gripper.



HUMAN IN THE LOOP:

Now in after dexnet return the possible grasp we want to evaluate them telling which are good which bad, 
to do that in DexNet grasp extract the operator assign an evaluation to the grasp, this evaluation is saved with the relative grasp.

DATASET:

composed by numpy arrays of dimension (4,100,100) where 4 are the channels(RGBD), the others two are the height and the width of the images.
The images are, as said, the picture with center in grasp center and orientation in such a way that the grasp lay on x axis.
The label is given by me and is based on what I consider good or not, this part is arbitrary so it influences a lot the results;

Some of the limits of this formulation are:
- For one scene I analyze several grasps and sometimes the grasp are very similar this could explain the overfitting.
- I'm not consistent in my labeling.
- More in general the grasps given by DexNet are usually good so it's difficult to understand if  a grasp is good or  is bad.

The MODEL:

I used a CNN model built using Keras.

Training and evaluating with the cubes and around 1000 images the net overfit probably because the cube are not a good example.

So I tryed with the paletta object and very less images and the results was better:

Validation loss: 0.003605542005971074
Validation accuracy: 1.0
Test loss: 0.4099232580350793
Test accuracy: 0.8695652122082917

Now I try with the hammer and with/without the variational encoder to see if it works better.


