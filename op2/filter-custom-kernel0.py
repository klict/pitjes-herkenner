import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('random.jpg')
kernel = np.full((3,3),-1,np.float32);
kernel[1][1] = 9;

dst = cv2.filter2D(img,-1,kernel)


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Custom kernel')
plt.xticks([]), plt.yticks([])
plt.show()
