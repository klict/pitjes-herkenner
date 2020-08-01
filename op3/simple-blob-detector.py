import cv2
import numpy as np

from op3.utils import apply_enhancement_and_segmentation

img = cv2.imread('../af1.1.jpg')

img = apply_enhancement_and_segmentation(img)

########################################
########################################
########################################
############ BLOB DETECTOR #############
########################################
########################################
########################################

# Set up the detector
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByCircularity = False

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img)

# Draw detected blobs as red circles.

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
