import cv2
import numpy as np
from utils import find_and_draw_contour, dominant_color_in_square, color_in_range, get_frame_from_image


def run_algoritme(img):
    # Our operations on the frame come here

    kernel = np.ones((5, 5), np.uint8)
    # img = cv2.imread('af1.jpg')

    # -----------------------------------------------
    # Using bilateralFilter to blur inside of object
    # while keeping outside edges
    # ------------------------------------------------
    blurred_img = cv2.bilateralFilter(img, 20, 50, 50)

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

    # -----------------------------------------
    # return result
    # -----------------------------------------

    walnut_condition = [25, 49, 92]
    dadel_condition = [24, 24, 33]
    gedroogd_condition = [20, 20, 20]
    kardemon_condition = [33, 75, 86]

    red = [255, 0, 0, ]
    green = [0, 255, 0]
    blue = [0, 0, 255]
    yellow = [255, 255, 0]
    white = [255, 255, 255]

    margin = 30

    for key_point in key_points:
        size = key_point.size
        x = int(key_point.pt[0])
        y = int(key_point.pt[1])

        try:

            dominant_color = dominant_color_in_square(modified_img, x, y, size)
            color = white

            if size > 150:
                if color_in_range(dominant_color, walnut_condition, margin):
                    color = red
                elif color_in_range(dominant_color, dadel_condition, margin):
                    color = blue
            else:
                if color_in_range(dominant_color, gedroogd_condition, margin):
                    color = yellow
                elif color_in_range(dominant_color, kardemon_condition, margin):
                    color = green

            img = find_and_draw_contour(img, x, y, size, color)
        except:
            print("x: " + str(x) + " y: " + str(y) + " size: " + str(size))

    return img
