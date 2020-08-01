import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('notenmix.JPG')
dst = cv2.Canny(img, 138,114,3);


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Canny')
plt.xticks([]), plt.yticks([])
plt.show()
