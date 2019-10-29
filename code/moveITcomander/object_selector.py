# import the necessary packages
import argparse
import cv2
import glob, os
import skimage.draw
import skimage.io
import numpy as np
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def find_contour(image):
    #image = cv2.imread(color_path)
    edged = cv2.Canny(image, 10, 250)
    #cv2.imshow("Edges", edged)
    #cv2.waitKey(0)
    
    #applying closing function 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Closed", closed)
    #cv2.waitKey(0)
    
    #finding_contours 
    #(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, cnts, _= cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    cv2.imshow("Output", image)
    cv2.waitKey(0)
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
#args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function


def build_manual_mask(color_path, mask_path):
    image = cv2.imread(color_path)
    #clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
    
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
    
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        black = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 0
        black[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]=255
        black= np.array(black).astype(np.uint8)
        #roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", black)
        cv2.waitKey(0)
        skimage.io.imsave(mask_path, black)
    
    # close all open windows
    cv2.destroyAllWindows()



if __name__ == '__main__':
    list_of_files = glob.glob('/home/lvianell/Desktop/Lorenzo_report/datasets/color/*')
    last_color_image= max(list_of_files, key=os.path.getctime)
    mask_path= last_color_image.replace("color", "mask")

    image = cv2.imread(last_color_image)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
    
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
    
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        black = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 0
        black[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]=255
        black= np.array(black).astype(np.uint8)
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        #find_contour(roi)     ##COULD BE A BETTER MASK...   
        cv2.imshow("ROI", black)
        cv2.waitKey(0)
        skimage.io.imsave(mask_path, black)
    
    # close all open windows
    cv2.destroyAllWindows()