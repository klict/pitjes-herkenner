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

    n_colors = 2
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
    d = int(r / 2)
    x1 = int(x - d)
    x2 = int(x + d)
    y1 = int(y - d)
    y2 = int(y + d)

    return img[y1:y2, x1:x2]


def replace_part_img_with_frame(img, frame, x, y, r):
    d = int(r / 2)
    x1 = int(x - d)
    x2 = int(x + d)
    y1 = int(y - d)
    y2 = int(y + d)

    img[y1:y2, x1:x2] = frame


def find_and_draw_contour(img, x, y, r, color):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = get_frame_from_image(gray, x, y, r)
    frame = get_frame_from_image(img, x, y, r)

    contours, _ = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours over original image
    cv2.drawContours(frame, contours, -1, color, 5)

    replace_part_img_with_frame(img, frame, x, y, r)

    return img
