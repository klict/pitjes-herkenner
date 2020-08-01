import cv2
import numpy as np


def dominant_color_in_square(img, x, y, r):
    # Making r smaller to ignore white part of image
    r -= 40

    x1 = int(x - r / 2)
    x2 = int(x + r / 2)
    y1 = int(y - r / 2)
    y2 = int(y + r / 2)

    frame = img[y1:y2, x1:x2]

    vectorized = frame.reshape((-1, 3))
    vectorized = np.float32(vectorized)

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(vectorized, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    # cv2.imshow(str(palette[np.argmax(counts)]) + "::size:" + str(r), cv2.resize(frame, (300, 300)))

    return palette[np.argmax(counts)]


def color_in_range(color, condition, margin):
    return (condition[0] - margin < color[0]) & (color[0] < condition[0] + margin) & \
           (condition[1] - margin < color[1]) & (color[1] < condition[1] + margin) & \
           (condition[2] - margin < color[2]) & (color[2] < condition[2] + margin)


def get_frame_from_image(img, x, y, r):
    x1 = int(x - r / 2)
    x2 = int(x + r / 2)
    y1 = int(y - r / 2)
    y2 = int(y + r / 2)

    return img[y1:y2, x1:x2]


def replace_part_img_with_frame(img, frame, x, y, r):
    x1 = int(x - r / 2)
    x2 = int(x + r / 2)
    y1 = int(y - r / 2)
    y2 = int(y + r / 2)

    img[y1:y2, x1:x2] = frame


def replace_color_in_image(img, color):
    height = img.shape[0]
    width = img.shape[1]

    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if img[y, x][0] != 255 & img[y, x][1] != 255 & img[y, x][2] != 255:
                img[y, x] = color
