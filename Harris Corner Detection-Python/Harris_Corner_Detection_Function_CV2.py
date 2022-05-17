import numpy as np
import cv2 as cv

img = cv.imread('chessboard.png')
cv.imshow('img', img)

double_float_img = np.float32(img)

gray = cv.cvtColor(double_float_img, cv.COLOR_BGR2GRAY)

dst = cv.cornerHarris(gray, 2, 3, 0.04)

dst = cv.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv.imshow('dst', img)

cv.waitKey(0)
cv.destroyAllWindows()