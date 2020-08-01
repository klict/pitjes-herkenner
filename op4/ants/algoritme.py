import cv2

af1_img = cv2.imread('af1.png')
af2_img = cv2.imread('af2.png')

af1_img_resized = cv2.resize(af1_img, (500, 500))
af2_img_resized = cv2.resize(af2_img, (500, 500))

# -----------------------------------------------
# Using gaussian to blur image
# ------------------------------------------------

af1_blurred_img = cv2.GaussianBlur(af1_img_resized, (5, 5), 0)
af2_blurred_img = cv2.GaussianBlur(af2_img_resized, (5, 5), 0)


result = cv2.bitwise_xor(af1_blurred_img, af2_blurred_img)

cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
