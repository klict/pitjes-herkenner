# import the necessary packages
import numpy as np
import cv2


def apply_enhancement_and_segmentation(img):
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
    img = cv2.dilate(img, kernel, iterations=5)
    img = cv2.erode(img, kernel, iterations=5)

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
    return res.reshape(img.shape)
