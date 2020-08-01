import cv2
from matplotlib import pyplot as plt

img = cv2.imread('af1.jpg')

blur = cv2.GaussianBlur(img, (9, 9), 0)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Gaus')
plt.xticks([]), plt.yticks([])
plt.show()
