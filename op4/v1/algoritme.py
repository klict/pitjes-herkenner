import cv2
import numpy as np
from cv2.cv2 import resizeWindow
from matplotlib import pyplot as plt

# -----------------------------------------------
# kernel for different filters use
# -----------------------------------------------
from utils import dominant_color_in_square, color_in_range, get_frame_from_image, replace_color_in_image, \
    replace_part_img_with_frame

kernel = np.ones((5, 5), np.uint8)

img = cv2.imread('af1.jpg')

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
eroded_img = cv2.erode(dilated_img, kernel, iterations=5)

# -----------------------------------------------
# Making background white to help segmentation phase
# ------------------------------------------------
modified_img = eroded_img.copy()

heightPixels = modified_img.shape[0]
widthPixels = modified_img.shape[1]

for x in range(0, widthPixels, 1):
    for y in range(0, heightPixels, 1):
        if modified_img[y, x][2] > 140:
            modified_img[y, x][0] = 255
            modified_img[y, x][1] = 255
            modified_img[y, x][2] = 255

# -----------------------------------------------
# using k-means to split image in 5 clusters
# 1: background, 2:dadel  3:walnut 4: dried something 5: kardemon
# @src #https://towardsdatascience.com/introduction-to-image-segmentation-with-k-means-clustering-83fd0a9e2fc3
# ------------------------------------------------
rgb_image = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)

# vectorizing
vectorized = rgb_image.reshape((-1, 3))
vectorized = np.float32(vectorized)

# criteria to be used as condition when to setup k-means algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# running k-means algorithm
K = 5
attempts = 30
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]

# applying label on image to retrieving final image
result_image = res.reshape(rgb_image.shape)

# -----------------------------------------------
# applying simple blob to retrieve all found items
# ------------------------------------------------

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByCircularity = False

detector = cv2.SimpleBlobDetector_create(params)

# found items
key_points = detector.detect(result_image)

image_with_key_points = cv2.drawKeypoints(result_image, key_points, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow("blobs detected", image_with_key_points)

# cv2.imshow("blobs detected", modified_img)

result_classification = {
    "dadel": 0,
    "walnut": 0,
    "gedroogd": 0,
    "kardemon": 0,
    "unknown": 0
}

walnut_condition = [25, 49, 92]
dadel_condition = [24, 24, 33]
gedroogd_condition = [20, 20, 20]
kardemon_condition = [33, 75, 86]

red = [255, 0, 0, ]
green = [0, 255, 0]
blue = [0, 0, 255]
yellow = [255, 255, 0]
black = [255, 255, 255]

margin = 30
for key_point in key_points:
    size = key_point.size
    x = int(key_point.pt[0])
    y = int(key_point.pt[1])

    dominant_color = dominant_color_in_square(modified_img, x, y, size)
    color = black

    if size > 180:
        if color_in_range(dominant_color, walnut_condition, margin):
            result_classification["walnut"] += 1
            color = red
        elif color_in_range(dominant_color, dadel_condition, margin):
            result_classification["dadel"] += 1
            color = blue
        else:
            result_classification["unknown"] += 1

    else:
        if color_in_range(dominant_color, gedroogd_condition, margin):
            result_classification["gedroogd"] += 1
            color = yellow
        elif color_in_range(dominant_color, kardemon_condition, margin):
            result_classification["kardemon"] += 1
            color = green
        else:
            print(dominant_color)
        result_classification["unknown"] += 1

    cv2.circle(resized_img, (x, y), 15, color, -1)

print(result_classification)

cv2.imshow("final", cv2.resize(resized_img, (500, 500)))

cv2.waitKey(0)
cv2.destroyAllWindows()
