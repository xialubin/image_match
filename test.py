import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('img_ref_cut.jpg')
img2 = cv.imread('img_tg_cut.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
# kp1 = sift.detect(gray1, None)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good[:10], None, flags=2)
plt.imshow(img3), plt.show()
