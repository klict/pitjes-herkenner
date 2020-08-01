# import the necessary packages
import numpy as np
import cv2

def apply_enhancement_and_segmentation(img):
    # -----------------------------------------------
    # kernel for different filters use
    # -----------------------------------------------
    kernel = np.ones((5, 5), np.uint8)


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
    modified_img = eroded_img

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
    return res.reshape(rgb_image.shape)
