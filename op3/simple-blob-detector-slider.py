import cv2
import numpy as np

from utils import apply_enhancement_and_segmentation

img = cv2.imread('../af1.1.jpg')

img = apply_enhancement_and_segmentation(img)

########################################
########################################
########################################
############ BLOB DETECTOR #############
########################################
########################################
########################################

# empty callback function for creating trackar
def callback(foo):
    pass


# create windows and trackbar
cv2.namedWindow('parameters')
cv2.createTrackbar('minThreshold', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
cv2.createTrackbar('maxThreshold', 'parameters', 0, 255, callback)  # change the maximum to whatever you like

# area
cv2.createTrackbar('filterByArea', 'parameters', 0, 1, callback)
cv2.createTrackbar('minArea', 'parameters', 0, 1000, callback)  # change the maximum to whatever you like
cv2.createTrackbar('maxArea ', 'parameters', 0, 1000, callback)  # change the maximum to whatever you like

# Circularity
cv2.createTrackbar('filterByCircularity ', 'parameters', 0, 1, callback)
cv2.createTrackbar('minCircularity ', 'parameters', 0, 100, callback)

# Filter by Convexity
cv2.createTrackbar('filterByConvexity', 'parameters', 0, 1, callback)
cv2.createTrackbar('minConvexity ', 'parameters', 0, 100, callback)

# Filter by Inertia (ratio of widest to thinnest point)
cv2.createTrackbar('filterByConvexity', 'parameters', 0, 1, callback)
cv2.createTrackbar('maxInertiaRatio ', 'parameters', 0, 100, callback)
cv2.createTrackbar('minConvexity ', 'parameters', 0, 100, callback)

while (True):

    params = cv2.SimpleBlobDetector_Params()

    # threshold
    if cv2.getTrackbarPos('minThreshold', 'parameters') & cv2.getTrackbarPos('minThreshold', 'parameters'):
        params.minThreshold = cv2.getTrackbarPos('minThreshold', 'parameters')
        params.maxThreshold = cv2.getTrackbarPos('minThreshold', 'parameters')

    # area
    params.filterByArea = cv2.getTrackbarPos('filterByArea', 'parameters') == 1
    if params.filterByArea:
        params.minArea = cv2.getTrackbarPos('minArea', 'parameters')
        params.maxArea = cv2.getTrackbarPos('maxArea', 'parameters')

    # Circularity
    params.filterByCircularity = cv2.getTrackbarPos('filterByCircularity', 'parameters') == 1
    if params.filterByCircularity:
        if cv2.getTrackbarPos('minCircularity', 'parameters') > 0:
            params.minCircularity = cv2.getTrackbarPos('minCircularity', 'parameters') / 100
        else:
            params.minCircularity = 0

    # Filter by Convexity
    params.filterByConvexity = cv2.getTrackbarPos('filterByConvexity', 'parameters') == 1
    if params.filterByConvexity:
        if cv2.getTrackbarPos('minConvexity', 'parameters') > 0:
            params.minConvexity = cv2.getTrackbarPos('minConvexity', 'parameters') / 100
        else:
            params.minConvexity = 0

    # Filter by Inertia (ratio of widest to thinnest point)
    params.filterByInertia = cv2.getTrackbarPos('filterByInertia', 'parameters') == 1

    if params.filterByInertia:
        if cv2.getTrackbarPos('maxInertiaRatio', 'parameters') > 0:
            params.maxInertiaRatio = cv2.getTrackbarPos('maxInertiaRatio', 'parameters') / 100
        else:
            params.maxInertiaRatio = 0

        if cv2.getTrackbarPos('minInertiaRatio', 'parameters') > 0:
            params.minInertiaRatio = cv2.getTrackbarPos('minInertiaRatio', 'parameters') / 100
        else:
            params.minInertiaRatio = 0

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", cv2.resize(im_with_keypoints, (900, 900)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
