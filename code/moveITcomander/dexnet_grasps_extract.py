import argparse
import json
import os
import time
import glob
import cv2

import matplotlib.pyplot as plt
plt.ion()

import numpy as np
import math

from PIL import ImageOps

import tkinter as tk

from autolab_core import RigidTransform, YamlConfig, Logger
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from visualization import Visualizer2D as vis

import sys
sys.path.insert(0, '/home/lvianell/Desktop/Lorenzo_report/gqcnn')


from gqcnn.grasping import RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw, FullyConvolutionalGraspingPolicySuction
from gqcnn.utils import GripperMode, NoValidGraspsException

#image="/home/lvianell/Desktop/Lorenzo_report/gqcnn/data/examples/single_object/depth_0.npy"
#image_1= "/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/depth_0.npy"

list_of_files = glob.glob('/home/lvianell/Desktop/Lorenzo_report/datasets/depth/*')
last_depth_image= max(list_of_files, key=os.path.getctime)

last_mask_path= last_depth_image.replace("depth", "mask").replace("npy", 'png')

last_color_path= last_depth_image.replace("depth", "color").replace("npy", 'png')

path_depth_png= last_depth_image.replace("depth", "depth_PNG").replace("npy", 'png')

path_grasp_position= last_depth_image.replace("depth", "grasp").replace("npy", 'txt')

if os.path.isfile(last_mask_path):
    mask= last_mask_path
else:
    mask=None

#mask= "/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/mask_0.png"
#mask=None

# set up logger
logger = Logger.get_logger('/home/lvianell/Desktop/Lorenzo_report/gqcnn/examples/policy_2.py')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Run a grasping policy on an example image')
    parser.add_argument('--model_name', type=str, default="GQCNN-4.0-PJ", help='name of a trained model to run')
    parser.add_argument('--depth_image', type=str, default=last_depth_image, help='path to a test depth image stored as a .npy file')
    parser.add_argument('--segmask', type=str, default=mask, help='path to an optional segmask to use')
    parser.add_argument('--camera_intr', type=str, default="/home/lvianell/Desktop/Lorenzo_report/gqcnn/data/calib/xtion/xtion.intr", help='path to the camera intrinsics')
    parser.add_argument('--model_dir', type=str, default="/home/lvianell/Desktop/Lorenzo_report/gqcnn/models", help='path to the folder in which the model is stored')
    parser.add_argument('--config_filename', type=str, default="/home/lvianell/Desktop/Lorenzo_report/gqcnn/cfg/examples/gqcnn_pj.yaml", help='path to configuration file to use')
    parser.add_argument('--fully_conv', action='store_true', help='run Fully-Convolutional GQ-CNN policy instead of standard GQ-CNN policy')
    args = parser.parse_args()
    model_name = args.model_name
    depth_im_filename = args.depth_image
    segmask_filename = args.segmask
    camera_intr_filename = args.camera_intr
    model_dir = args.model_dir
    config_filename = args.config_filename
    fully_conv = args.fully_conv
    print fully_conv

    assert not (fully_conv and depth_im_filename is not None and segmask_filename is None), 'Fully-Convolutional policy expects a segmask.'

    if depth_im_filename is None:
        if fully_conv:
            depth_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             '..',
                                             'data/examples/clutter/primesense/depth_0.npy')
        else:
            depth_im_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             '..',
                                             'data/examples/single_object/primesense/depth_0.npy')
    if fully_conv and segmask_filename is None:
        segmask_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        '..',
                                        'data/examples/clutter/primesense/segmask_0.png')
    if camera_intr_filename is None:
        camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            '..',
                                            'data/calib/xtion/xtion.intr')    
   

    # set model if provided 
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '../models')
    model_path = os.path.join(model_dir, model_name)

    # get configs
    model_config = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    try:
        gqcnn_config = model_config['gqcnn']
        gripper_mode = gqcnn_config['gripper_mode']
    except:
        gqcnn_config = model_config['gqcnn_config']
        input_data_mode = gqcnn_config['input_data_mode']
        if input_data_mode == 'tf_image':
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == 'tf_image_suction':
            gripper_mode = GripperMode.LEGACY_SUCTION                
        elif input_data_mode == 'suction':
            gripper_mode = GripperMode.SUCTION                
        elif input_data_mode == 'multi_suction':
            gripper_mode = GripperMode.MULTI_SUCTION                
        elif input_data_mode == 'parallel_jaw':
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError('Input data mode {} not supported!'.format(input_data_mode))
    
    # set config
    if config_filename is None:
        if gripper_mode == GripperMode.LEGACY_PARALLEL_JAW or gripper_mode == GripperMode.PARALLEL_JAW:
            if fully_conv:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/fc_gqcnn_pj.yaml')
            else:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/gqcnn_pj.yaml')
        elif gripper_mode == GripperMode.LEGACY_SUCTION or gripper_mode == GripperMode.SUCTION:
            if fully_conv:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/fc_gqcnn_suction.yaml')
            else:
                config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '..',
                                               'cfg/examples/gqcnn_suction.yaml')
    #print config_filename
    # read config
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config['inpaint_rescale_factor']
    policy_config = config['policy']

    # make relative paths absolute
    if 'gqcnn_model' in policy_config['metric'].keys():
        policy_config['metric']['gqcnn_model'] = model_path
        if not os.path.isabs(policy_config['metric']['gqcnn_model']):
            policy_config['metric']['gqcnn_model'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                  '..',
                                                                  policy_config['metric']['gqcnn_model'])
            
    # setup sensor
    camera_intr = CameraIntrinsics.load(camera_intr_filename)
        
    # read images
    depth_data = np.load(depth_im_filename)
    c_image= np.array(cv2.imread(last_color_path))
    #print depth_data
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    #color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
    #                      frame=camera_intr.frame)
    color_im = ColorImage(c_image, frame=camera_intr.frame)
    # optionally read a segmask
    segmask = None
    if segmask_filename is not None:
        segmask = BinaryImage.open(segmask_filename)
        print "SegMask Opening __________________"
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    if segmask is None:
        segmask = valid_px_mask
    else:
        segmask = segmask.mask_binary(valid_px_mask)
    
    # inpaint
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
        
    if True :#'input_images' in policy_config['vis'].keys() and policy_config['vis']['input_images']:
        vis.figure(size=(10,10))
        num_plot = 1
        if segmask is not None:
            num_plot = 2
        vis.subplot(1,num_plot,1)
        vis.imshow(depth_im)
        if segmask is not None:
            vis.subplot(1,num_plot,2)
            vis.imshow(segmask)
        vis.show()
        

    flag_mask="Y"#raw_input("Do you want to use mask? [Y/n]")

    if flag_mask=="n":
        print "------ YOU CHOOSE WITHOUT MASK -----"
        segmask=None
    else:
        print "------ YOU CHOOSE TO USE THE MASK -----"

    # create state
    #segmask=None
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    # set input sizes for fully-convolutional policy
    if fully_conv:
        policy_config['metric']['fully_conv_gqcnn_config']['im_height'] = depth_im.shape[0]
        policy_config['metric']['fully_conv_gqcnn_config']['im_width'] = depth_im.shape[1]

    # init policy
    if fully_conv:
        #TODO: @Vishal we should really be doing this in some factory policy
        if policy_config['type'] == 'fully_conv_suction':
            policy = FullyConvolutionalGraspingPolicySuction(policy_config)
        elif policy_config['type'] == 'fully_conv_pj':
            policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
        else:
            raise ValueError('Invalid fully-convolutional policy type: {}'.format(policy_config['type']))
    else:
        policy_type = 'cem'
        if 'type' in policy_config.keys():
            policy_type = policy_config['type']
        if policy_type == 'ranking':
            policy = RobustGraspingPolicy(policy_config)
        elif policy_type == 'cem':
            policy = CrossEntropyRobustGraspingPolicy(policy_config)
        else:
            raise ValueError('Invalid policy type: {}'.format(policy_type))



    pathDataset="/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/"

    #creo dataset:
    pngCounter = 0#len(glob.glob1(pathDataset,"*.txt"))

    # save depth image in /home/lvianell/Desktop/gqcnn/data/dataset_human_demostration
    vis.figure(size=(10,10))
    vis.imshow(rgbd_im.depth)#,
                #vmin=policy_config['vis']['vmin'],
                #vmax=policy_config['vis']['vmax'])
    #vis.savefig(pathDataset+"depth_im_"+str(pngCounter))
    vis.savefig(path_depth_png)
    vis.title("DepthImage")
    vis.show()

    print path_grasp_position
    text_file = open(path_grasp_position, "w") #open(pathDataset+"grasps_"+str(pngCounter)+".txt", "w")
    text_file.write("[center_x, center_y]"+"\t"+"angle"+"\t"+"depth"+"\t"+"width"+"\t"+"Qvalue"+ "\t"+"human_evaluation"+"\n")


    flag_human="y"#raw_input("Do you want to use human contribute? [y/N]")
    # query policy
    policy_start = time.time()


    if flag_human=="y":

        print "You choose human demostration"

        evaluations=[]
        grasps, q_values = policy.action_set(state)
        print len(grasps)

        #vis.figure(size=(10,10))
        #vis.imshow(rgbd_im.color, vmin=policy_config['vis']['vmin'], vmax=policy_config['vis']['vmax'])

        rgb_image= cv2.imread(last_color_path)

        d_image= np.array(rgbd_im.depth.data)
        #print rgbd_im.depth.shape

        #TODO :BUILD BOTTON

        for g, q in zip(grasps, q_values):
            print g.center, g.width, g.angle
            #print q
            if True:

                vis.figure()#size=(10,10))
                vis.imshow(rgbd_im.depth)#,vmin=policy_config['vis']['vmin'],vmax=policy_config['vis']['vmax'])
                vis.grasp(g, scale=1, show_center=True, show_axis=False)
                #vis.savefig("/home/lvianell/Desktop/Lorenzo_report/datasets/grasp_figures/test.png", bbox_inches='tight', pad_inches=0)
                vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(g.depth, q))
                vis.show()
                
                #grasp_image = cv2.imread("/home/lvianell/Desktop/Lorenzo_report/datasets/grasp_figures/test.png",0)
                rows,cols, canal = rgb_image.shape
                DIM= 50    #diminsion resulting matrix= DIM*2
                LEN_GRASP= 20
                M = cv2.getRotationMatrix2D((int(g.center.x),int(g.center.y)),math.degrees(g.angle),1)
                
                dst = cv2.warpAffine(rgb_image,M,(cols,rows))
                #cv2.line(dst,(int(g.center.x)-LEN_GRASP,int(g.center.y)),(int(g.center.x)+LEN_GRASP,int(g.center.y)),(255,255,255),1)   
                dst= dst[int(g.center.y)-DIM:int(g.center.y)+DIM, int(g.center.x)-DIM:int(g.center.x)+DIM]
                
                print dst.shape

                
                
                dst_d= cv2.warpAffine(d_image,M,(cols,rows))
                #cv2.line(dst_d,(int(g.center.x)-LEN_GRASP,int(g.center.y)),(int(g.center.x)+LEN_GRASP,int(g.center.y)),(1,0,0),1)
                dst_d= dst_d[int(g.center.y)-DIM:int(g.center.y)+DIM, int(g.center.x)-DIM:int(g.center.x)+DIM]

                print dst_d.shape

                result= np.dstack((dst/255.0, dst_d))   # in this way the results are alredy normalized

                print result.shape

                eval= raw_input("How do you think to evaluate this grasp? [0=bad/1=good]")
                evaluations.append(eval)
                text_file.write(str(g.center)+"\t"+str(g.angle)+"\t"+str(g.depth)+"\t"+str(g.width)+"\t"+str(q)+ "\t"+str(eval)+"\n")
                

                plot=True
                if plot==True:
                    plt.figure()
                    plt.subplot(1,2,1)
                    plt.axis("off")
                    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
                    plt.subplot(1,2,2)
                    plt.axis("off")
                    plt.imshow(dst_d, cmap=plt.cm.gray_r)#, vmin=policy_config['vis']['vmin'], vmax=policy_config['vis']['vmax'])

                t = time.localtime()
                timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
                train_set_path= "/home/lvianell/Desktop/Lorenzo_report/datasets/grasp_figures/paletta2"
                test_set_path="/home/lvianell/Desktop/Lorenzo_report/datasets/grasp_figures/paletta2/test_set"
                data_set_path=test_set_path 
                if eval== "1":
                    if plot==True:
                        plt.savefig(data_set_path+"/good/"+str(timestamp)+".png")
                        plt.title("GRASP EVALUATED AS GOOD")
                        plt.ion()
                        plt.show()
                        #plt.pause(0.01)
                        plt.close('all')
                    np.save(data_set_path+"/good/"+str(timestamp)+".npy", result)
                elif eval=="0":
                    if plot==True:
                        plt.savefig(data_set_path+"/bad/"+str(timestamp)+".png")
                        plt.title("GRASP EVALUATED AS BAD")
                        plt.show()
                        plt.close("all") 
                    np.save(data_set_path+"/bad/"+str(timestamp)+".npy", result)   
                else:
                    print "SKIPPED"
                    plt.close("all") 

        # save txt file containing the grasps informations and relatiive evaluations
        if False:
            max_evaluated=0 ; max_evaluated_grasp=None
            for g, q, e in zip(grasps, q_values, evaluations):
                if float(e)*float(q)>max_evaluated:
                    max_evaluated= float(e)*float(q)
                    max_evaluated_grasp= g
                #text_file.write(str(g.center)+"\t"+str(g.angle)+"\t"+str(g.depth)+"\t"+str(g.width)+"\t"+str(q)+ "\t"+str(e)+"\n")
            #text_file.close()

            best_grasp=max_evaluated_grasp
            best_q= max_evaluated

            print "center of the grasp in pixels coordinates"
            print best_grasp.center
            print "orientation of the grasp"
            print best_grasp.angle
            print " depth grasp"
            print best_grasp.depth

            RATIO_PIXEL_M= 386
            x_position= -(221-best_grasp.center.y)/RATIO_PIXEL_M
            y_position= -(553- best_grasp.center.x)/RATIO_PIXEL_M
            print "center of the grasp in meters"
            print x_position, y_position

            logger.info('Planning took %.3f sec' %(time.time() - policy_start))

            # vis final grasp
            if policy_config['vis']['final_grasp']:
                vis.figure(size=(10,10))
                vis.imshow(rgbd_im.depth)#,
                        #vmin=policy_config['vis']['vmin'],
                        #vmax=policy_config['vis']['vmax'])
                vis.grasp(best_grasp, scale=2.5, show_center=True, show_axis=False)
                vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(best_grasp.depth, best_q))
                vis.show()

            flag_eval=raw_input("Do you like this grasp? [Y/n]")

            if flag_eval=="n":
                text_file.write(str(best_grasp.center)+"\t"+str(best_grasp.angle)+"\t"+str(best_grasp.depth)+"\t"+str(best_grasp.width)+"\t"+str(best_q)+ "\t"+str(0)+"\n")
                print "------ REPEAT DEXNET -----"
            else:
                print "------ OK THEN I SAVE THE GRASP-----"
                done= True
                text_file.write(str(best_grasp.center)+"\t"+str(best_grasp.angle)+"\t"+str(best_grasp.depth)+"\t"+str(best_grasp.width)+"\t"+str(best_q)+ "\t"+str(1)+"\n")


    else:
        done = False 

        while(not(done)):

            print "You choose autonoumus"

            action = policy(state)

            print "center of the grasp in pixels coordinates"
            print action.grasp.center
            print "orientation of the grasp"
            print action.grasp.angle
            print " depth grasp"
            print action.grasp.depth

            RATIO_PIXEL_M= 386
            x_position= -(221-action.grasp.center.y)/RATIO_PIXEL_M
            y_position= -(553- action.grasp.center.x)/RATIO_PIXEL_M
            #print "center of the grasp in meters"
            #print x_position, y_position

            logger.info('Planning took %.3f sec' %(time.time() - policy_start))

            # vis final grasp
            if policy_config['vis']['final_grasp']:
                vis.figure(size=(10,10))
                vis.imshow(rgbd_im.color,
                            vmin=policy_config['vis']['vmin'],
                            vmax=policy_config['vis']['vmax'])
                vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
                vis.title('Planned grasp at depth {0:.3f}m with Q={1:.3f}'.format(action.grasp.depth, action.q_value))
                vis.show()

            flag_eval=raw_input("Do you like this grasp? [Y/n]")

            if flag_eval=="n":
                text_file.write(str(action.grasp.center)+"\t"+str(action.grasp.angle)+"\t"+str(action.grasp.depth)+"\t"+str(action.grasp.width)+"\t"+str(action.q_value)+ "\t"+str(0)+"\n")
                print "------ REPEAT DEXNET -----"
            else:
                print "------ OK THEN I SAVE THE GRASP-----"
                done= True
                text_file.write(str(action.grasp.center)+"\t"+str(action.grasp.angle)+"\t"+str(action.grasp.depth)+"\t"+str(action.grasp.width)+"\t"+str(action.q_value)+ "\t"+str(1)+"\n")
