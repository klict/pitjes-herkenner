import cv2
import numpy as np
from matplotlib import pyplot as plt

kernel = np.ones((5, 5), np.uint8)
img = cv2.imread('af1.jpg',0)
#img = cv2.imread('notenmix.JPG')

# -----------------------------------------------
# resizing img to 900 by 900 for faster processing
# -----------------------------------------------
resized_img = cv2.resize(img, (900, 900))

# -----------------------------------------------
# Using bilateralFilter to blur inside of object
# while keeping outside edges
# ------------------------------------------------
blurred_img = cv2.bilateralFilter(resized_img, 20, 50, 50)

# -----------------------------------------------
# removing noise from image
# ------------------------------------------------
clean_img = cv2.morphologyEx(blurred_img, cv2.MORPH_OPEN, kernel)

# -----------------------------------------------
# dilating 5 times to disconnect any connected objects
# eroding 5 times to return image as close to original, while keeping certain changes
# like disconnected objects
# ------------------------------------------------
dilated_img = cv2.dilate(clean_img, kernel, iterations=5)
img = cv2.erode(dilated_img, kernel, iterations=5)

## testing
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
