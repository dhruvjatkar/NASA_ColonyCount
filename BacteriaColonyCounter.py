#python BacteriaColonyCounter.py -i [imagefile] -o [imagefile]

 import argparse
import imutils
import cv2
import numpy as np

#arguement parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-o", "--output", required=True,
    help="path to the output image")
args = vars(ap.parse_args())
  
counter = {}
 
image_orig = cv2.imread(args["image"])
height_orig, width_orig = image_orig.shape[:2]
 
image_contours = image_orig.copy()
 
#input whichever colors you desire
colors = ['white']
for color in colors:
 
    image_to_process = image_orig.copy()
 
    #initialize counter
    counter[color] = 0
 
    image_to_process = (255-image_to_process)
    lower = np.array([ 50,  50,  40])
    upper = np.array([100, 120,  80])
 
    image_mask = cv2.inRange(image_to_process, lower, upper)
    image_res = cv2.bitwise_and(image_to_process, image_to_process, mask = image_mask)
 
    #convert image to grayscale, blur
    image_gray = cv2.cvtColor(image_res, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
 

    image_edged = cv2.Canny(image_gray, 50, 100)
    image_edged = cv2.dilate(image_edged, None, iterations=1)
    image_edged = cv2.erode(image_edged, None, iterations=1)
 
    cnts = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
    #loop over contours
    for c in cnts:
         
        if cv2.contourArea(c) < (1/100):
            continue
         
        hull = cv2.convexHull(c)
        if color == 'blue':
            cv2.drawContours(image_contours,[hull],0,(0,0,255),1)
        elif color == 'white':
            # prints contours in green color
            cv2.drawContours(image_contours,[hull],0,(0,255,0),1)
 
        counter[color] += 1

#print labeled colonies by  color
    print("{} {} colonies".format(counter[color],color))

#output image
cv2.imwrite(args["output"],image_contours)
