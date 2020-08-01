import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils

img = cv2.resize(cv2.imread('../af1.1.jpg'), (900, 900))

heightPixels = img.shape[0]
widthPixels = img.shape[1]

# afbeelding naar kleine formaat veranderd om te tonen
# cv2.imshow("mix van objecten", img)

# enhancement
# vervagen
img = cv2.bilateralFilter(img, 20, 50, 50)

# cv2.imshow("enhancement", img)

# segmentatie

## segmantie - verbeteringen
kernel = np.ones((5, 5), np.uint8)

##ruis weggehaald
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

### dilate en erode
img = cv2.dilate(img, kernel,iterations=4)
img = cv2.erode(img, kernel,iterations=4)

# R(GB) waarde van onder 100 omzetten naar 0
for x in range(0, widthPixels, 1):
    for y in range(0, heightPixels, 1):
        if img[y, x][2] > 140:
            img[y, x][0] = 255
            img[y, x][1] = 255
            img[y, x][2] = 255


# cv2.imshow("rgb waarde aangepast", img)
## segmentatie met k-means


# k-means
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1, 3))
vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 5
attempts = 10
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape(img.shape)

# cv2.imshow( 'Segmented Image when K = %i', cv2.resize(result_image, (900, 900)) )

figure_size = 15
plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()
