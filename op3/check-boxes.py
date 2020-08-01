import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng

img = cv2.imread('../af1.1.jpg')
#img = cv2.imread('notenmix.JPG')

# afbeelding naar kleine formaat veranderd om te tonen
cv2.imshow( "mix van objecten", cv2.resize(img, (900, 900)) )

# enhancement
img = cv2.bilateralFilter(img,20,50,50)

cv2.imshow( "enhancement", cv2.resize(img, (900, 900)) )


# segmentatie
kernel = np.ones((5,5),np.uint8)

#Removing noise
img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)

# color thresholding

cv2.imshow( "segmentatie", cv2.resize(img, (900, 900)) )


## checking

def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(img, threshold, threshold * 2,0,0)
    # Find contours
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    # Show in a window
    cv2.imshow('Contours', cv2.resize(drawing, (900, 900)))

# Create Window
source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, cv2.resize(img, (900, 900)))


max_thresh = 255
thresh = 100 # initial threshold
cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)


cv2.waitKey(0)
cv2.destroyAllWindows()
