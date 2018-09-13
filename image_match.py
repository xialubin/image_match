import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def get_sift_feature(image1, image2):
    """img1 and img2 must be gray image"""

    sift = cv.xfeatures2d.SIFT_create()
    # kp1 = sift.detect(gray1, None)
    k_p1, des1 = sift.detectAndCompute(image1, None)
    k_p2, des2 = sift.detectAndCompute(image2, None)
    # img_draw = cv.drawKeypoints(gray1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imwrite('sift_keypoints_1.jpg', img_draw)

    bf = cv.BFMatcher()
    match = bf.knnMatch(des1, des2, k=2)
    return match, k_p1, k_p2


def feature_match(match, k_p1, k_p2):
    good = []
    mask = []
    for m, n in match:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    # cv.drawMatchesKnn expects list of lists as matches.
    if len(good) > 4:
        ptsA = np.float32([k_p1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([k_p2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        ransacReprojthreshold = 5
        mask, status = cv.findHomography(ptsA, ptsB, cv.RANSAC, ransacReprojthreshold)
    return mask


def image_select(image_bb, image_or):
    """
    select a rectangle part of image
    :param image_bb: the image with black border
    :param image_or: the original image
    :return: the same rectangle part of image_bb and image_or
    """
    gray_bb = np.int32(cv.cvtColor(image_bb, cv.COLOR_BGR2GRAY))
    gray_bb[gray_bb != 0] = 1

    h = gray_bb.shape[0]
    w = gray_bb.shape[1]
    index_up, index_down, index_left, index_right = 0, h - 1, 0, w - 1
    num_up, num_down, num_left, num_right = 0, 0, 0, 0

    while index_up < h:
        if sum(gray_bb[index_up, :]) != 0:
            break
        index_up += 1
        print('index_up:', index_up)
    while index_down >= 0:
        if sum(gray_bb[index_down, :]) != 0:
            break
        index_down -= 1
        print('index_down:', index_down)
    while index_left < w:
        if sum(gray_bb[:, index_left]) != 0:
            break
        index_left += 1
        print('index_left:', index_left)
    while index_right >= 0:
        if sum(gray_bb[:, index_right]) != 0:
            break
        index_right -= 1
        print('index_right:', index_right)

    while index_up < h and index_down >= 0 and index_left < w and index_right >= 0:
        flag = 4

        if sum(gray_bb[index_up, :]) < (index_right - index_left + 1):
            index_up += 1
            print('index_up:', index_up)
            flag -= 1
        if sum(gray_bb[index_down, :]) < (index_right - index_left + 1):
            index_down -= 1
            print('index_down:', index_down)
            flag -= 1
        if sum(gray_bb[:, index_left]) < (index_down - index_up + 1):
            index_left += 1
            print('index_left:', index_left)
            flag -= 1
        if sum(gray_bb[:, index_right]) < (index_down - index_up + 1):
            index_right -= 1
            print('index_right:', index_right)
            flag -= 1

        if flag == 4:
            break

    img_cut1 = image_bb[index_up:index_down, index_left:index_right, :]
    img_cut2 = image_or[index_up:index_down, index_left:index_right, :]

    return img_cut1, img_cut2


# def image_cut(image_black_border, image_or):
#     """remove the black border of image"""
#     gray_bb = np.float32(cv.cvtColor(image_black_border, cv.COLOR_BGR2GRAY))
#     # gray_or = np.float32(cv.cvtColor(image_or, cv.COLOR_BGR2GRAY))
#     img_cut1 = []
#     img_cut2 = []

#     # binary = gray_bb
#     # binary[binary != 0] = 255.0

#     dst = cv.cornerHarris(gray_bb, 2, 3, 0.04)

#     # dst = cv.dilate(dst, None)
#     #
#     # # Threshold for an optimal value, it may vary depending on the image.
#     # image_black_border[dst >= 0.01*dst.max()] = [0, 0, 255]
#     #
#     # # cv.imwrite('test.jpg', image_black_border)
#     #
#     # cv.imshow('dst', image_black_border)
#     # if cv.waitKey(0) & 0xff == 27:
#     #     cv.destroyAllWindows()

#     corner_p = np.where(dst == dst.max())
#     index_x = corner_p[0].item()
#     index_y = corner_p[1].item()

#     if gray_bb[index_x - 2, index_y - 2] == 0:
#         img_cut1 = image_black_border[corner_p[0].item():, corner_p[1].item():, :]
#         img_cut2 = image_or[corner_p[0].item():, corner_p[1].item():, :]
#     elif gray_bb[index_x + 2, index_y - 2] == 0:
#         img_cut1 = image_black_border[:corner_p[0].item(), corner_p[1].item():, :]
#         img_cut2 = image_or[:corner_p[0].item(), corner_p[1].item():, :]
#     elif gray_bb[index_x - 2, index_y + 2] == 0:
#         img_cut1 = image_black_border[corner_p[0].item():, :corner_p[1].item(), :]
#         img_cut2 = image_or[corner_p[0].item():, :corner_p[1].item(), :]
#     elif gray_bb[index_x + 2, index_y + 2] == 0:
#         img_cut1 = image_black_border[:corner_p[0].item(), :corner_p[1].item(), :]
#         img_cut2 = image_or[:corner_p[0].item(), :corner_p[1].item(), :]
#     else:
#         print('can not fine the corner!')

#     return img_cut1, img_cut2


img1 = cv.imread('2015_00001.png')  # query image: constant
img2 = cv.imread('2015_00001_1.png')  # train image: variable
# img1 = cv.imread('image1.jpg')  # query image: constant
# img2 = cv.imread('image2.jpg')  # train image: variable
# img1 = cv.imread('0000_nir.tiff')  # query image: constant
# img2 = cv.imread('0000_rgb.tiff')  # train image: variable
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
matches, kp1, kp2 = get_sift_feature(gray1, gray2)
H = feature_match(matches, kp1, kp2)

imgOut = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]), borderMode=False,
                            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

cv.imwrite('imgout.jpg', imgOut)

image_cut1, image_cut2 = image_select(imgOut, img1)
cv.imwrite('image_cut1.jpg', image_cut1)
cv.imwrite('image_cut2.jpg', image_cut2)
