import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# img1 = cv.imread('2015_00001.png')
# img2 = cv.imread('2015_00001_1.png')
# img1 = cv.imread('0000_nir.tiff')
# img2 = cv.imread('0000_rgb.tiff')
img1 = cv.imread('image1.jpg')
img2 = cv.imread('image2.jpg')
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
# kp1 = sift.detect(gray1, None)
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
# img_draw = cv.drawKeypoints(gray1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imwrite('sift_keypoints_1.jpg', img_draw)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
# cv.drawMatchesKnn expects list of lists as matches.
if len(good) > 4:
    ptsA = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    H, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojThreshold)
    imgOut = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]), borderMode=False, flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

    allImg = np.concatenate((img1, img2, imgOut), axis=1)
    # plt.title('result')
    # plt.imshow(allImg)
    # plt.show()
    cv.imwrite('result.jpg', allImg)
    cv.imwrite('imgout.jpg', imgOut)


# img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

