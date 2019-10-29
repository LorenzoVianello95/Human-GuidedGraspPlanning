# Program that take the rgb image and return the mask of the  reds cubes

import sys
sys.path.insert(0, '/home/lvianell/Desktop/Lorenzo_report/code_mask_NN/Mask_RCNN')

sys.path.insert(0,'/home/lvianell/Desktop/Lorenzo_report/code_mask_NN/Mask_RCNN/samples/balloon')


from balloon_modified_works import BalloonConfig, BalloonDataset, train
from mrcnn.config import Config
from mrcnn import model as modellib, utils



"""
to run :

python3 build_mask.py 

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.io
import glob
import os

pathDataset="/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/"
txtCounter = 0#len(glob.glob1(pathDataset,"*.txt"))

list_of_files = glob.glob('/home/lvianell/Desktop/Lorenzo_report/datasets/color/*')
last_color_image= max(list_of_files, key=os.path.getctime)
mask_path= last_color_image.replace("color", "mask")


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    black = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 0
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, 0).astype(np.uint8)
        splash = np.where(np.logical_not(mask), splash, 255).astype(np.uint8)
        
        #print(splash)
    else:
        splash = black.astype(np.uint8)+255     #se non trova niente metto tutto bianco in maniera tale che comunque esegua un grasp...
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = mask_path #"/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/mask_"+str(txtCounter)+".png"
        skimage.io.imsave(file_name, splash)
    
    print("Saved to ", file_name)




command= "splash"
image_path= last_color_image#"/home/lvianell/Desktop/Lorenzo_report/datasets/dexnet_maskcnn_human/color_"+str(txtCounter)+".png"
logs_path= "/home/lvianell/Desktop/Lorenzo_report/code_mask_NN/Mask_RCNN/logs"
weights_path="/home/lvianell/Desktop/Lorenzo_report/code_mask_NN/Mask_RCNN/mask_rcnn_balloon_0030.h5"

# Configurations
class InferenceConfig(BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model
model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=logs_path)

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

detect_and_color_splash(model, image_path= image_path, video_path=None)

