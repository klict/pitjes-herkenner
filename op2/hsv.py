import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('af1.jpg')
hsvImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);


# empty callback function for creating trackar
def callback(foo):
    pass
cv2.namedWindow('parameters')

cv2.createTrackbar('Hue', 'parameters', 0, 179, callback)  # change the maximum to whatever you like
cv2.createTrackbar('Saturation', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
cv2.createTrackbar('Value', 'parameters', 0,255, callback)

while(True):
    hsvImg[:,:,0] = cv2.getTrackbarPos('Hue', 'parameters')
    hsvImg[:,:,1] = cv2.getTrackbarPos('Saturation', 'parameters')
    hsvImg[:,:,2] = cv2.getTrackbarPos('Value', 'parameters')*2+3

    imgBack = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR);
    imgResized = cv2.resize(imgBack, (960, 540))                    # Resize image

    cv2.imshow('hsv', imgResized)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cv2.destroyAllWindows()
