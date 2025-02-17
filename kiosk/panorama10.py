
import numpy as np
import imutils
import cv2
import os
from os import listdir, mkdir
from os.path import isfile, join, exists
from utils import load_image
from transform import *
import math

import datetime

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.reinforce_rows = 100
        self.meaningful_threshold = 0.7
        self.homo = None
        self.ptsA = None
        self.ptsB = None

    def stitch(self, images, ratio=0.6, reprojThresh=4.0, calibrated=False):

        def getError(self, f1, f2):

            self.l_kps = self.spherical_calibration(self.image_left.shape[:2], self.l_ori_kps, f1)
            self.r_kps = self.spherical_calibration(self.image_right.shape[:2], self.r_ori_kps, f2)
            # print 'Calibration done'

            # k1 = self.drawKps(self.image_left, self.l_ori_kps)
            # print len(self.l_ori_kps)
            # cv2.imshow('original keypoints', k1)

            # k2 = self.drawKps(to_spherical(self.image_left, f), self.l_kps)
            # print len(self.l_kps)
            # cv2.imshow('sphericalized keypoints', k2)

            # print 'self.l_kps:', self.l_kps
            # print 'self.r_kps:', self.r_kps

            # cv2.waitKey()

            # tmp_image_left = to_spherical(self.image_left, f)
            # tmp_image_right = to_spherical(self.image_right, f)
            # vis = self.drawMatches(tmp_image_left, tmp_image_right, self.l_kps, self.r_kps, self.matches, self.status)
            # cv2.imshow('matches', vis)
            # cv2.waitKey()
            # exit(0)

            self.ptsA = np.float32([self.l_kps[i] for (i, _) in self.matches])

            self.ptsB = np.float32([self.r_kps[i] for (_, i) in self.matches])

            # self.drawKps(to_fish_eye(self.image_left, deg_), self.l_kps)

            (H, status) = cv2.findHomography(self.ptsB, self.ptsA, cv2.RANSAC,
                                             reprojThresh)
            # error = abs(H[0, 1]) / 2000. + abs(H[1, 0]) / 500. + abs(H[2, 0]) + abs(H[2, 1]) / 10.
            # print 'ptsA:', self.ptsA
            # print 'ptsB:', self.ptsB
            # print H
            absH = np.core.abs(np.copy(H))

            error = np.zeros(4)
            error[0] = 0 if absH[0, 1] < 0.02 else np.power((absH[0, 1] - 0.02) / 2000., 2)
            error[1] = 0 if absH[1, 0] < 0.005 else np.power((absH[1, 0] - 0.005) / 1000, 2)
            error[2] = 0 if absH[2, 0] < 0.00001 else np.power(absH[2, 0] - 0.00001, 2)
            error[3] = 0 if absH[2, 1] < 0.0001 else np.power((absH[2, 1] - 0.0001) / 10., 2)

            return (sum(error), H)

        # unpack the images
        (self.image_left, self.image_right) = images

        (kps, features) = self.detectAndDescribe(self.image_left)
        self.l_ori_kps = np.copy(kps)
        self.l_kps = np.copy(kps)
        self.l_features = np.copy(features)

        (kps, features) = self.detectAndDescribe(self.image_right)
        self.r_ori_kps = np.copy(kps)
        self.r_kps = np.copy(kps)
        self.r_features = np.copy(features)

        matchResult = self.matchKeypoints(self.r_kps, self.l_kps,
                                          self.r_features, self.l_features, ratio, reprojThresh)

        if matchResult is None:
            print 'images are not match!'
            return np.hstack((self.image_left, self.image_right))

        (self.matches, H, self.status) = matchResult
        # print 'self.matches:', self.matches

        # vis = self.drawMatches(self.image_left, self.image_right, self.l_ori_kps, self.r_ori_kps, self.matches, self.status)
        # cv2.imshow('matches_', vis)

        if not calibrated:
            min_f = min(images[0].shape[0], images[0].shape[1])
            max_f = max(images[0].shape[0], images[0].shape[1])
        else:
            min_f = max(images[0].shape[0], images[0].shape[1]) * 2
            max_f = max(images[0].shape[0], images[0].shape[1]) * 3

        print 'min_f, max_f:', min_f, max_f

        n_f_try = 20
        f_to_try = np.linspace(min_f, max_f, n_f_try)

        self.coolest = [999, None, 0, 0]

        for idx1, fcl1 in enumerate(f_to_try):
            for fcl2 in f_to_try[max(0, idx1 - 3):min(idx1 + 3, n_f_try)]:
                (now_error, self.homo) = getError(self, fcl1, fcl2)
                if now_error < self.coolest[0]:
                    self.coolest = [now_error, self.homo, fcl1, fcl2]

        self.best_f = self.coolest[2:]
        self.homo = self.coolest[1]
        self.homo[0, 1] = self.homo[1, 0] = self.homo[2, 0] = self.homo[2, 1] = 0
        if self.homo[0, 2] <= 0:
            return self.image_left if self.image_left.shape[1] > self.image_right.shape[1] else self.image_right

        image_left = to_spherical(self.image_left, self.best_f[0])
        image_right = to_spherical(self.image_right, self.best_f[1])

        image_left = self.trimSurplusCols(image_left)
        image_right = self.trimSurplusCols(image_right)

        image_right_wrapped = cv2.warpPerspective(image_right, self.homo,
                                         (image_left.shape[1] + int(self.homo[0, 2]),
                                          image_left.shape[0]),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print 'best f:', self.best_f
        print 'homo: ', self.homo

        result = self.paste_4(image_left, image_right_wrapped, moved=self.homo[0, 2])
        result = self.trimSurplusCols(result)

        return result

    def ellipse_calibration(self, shape, kps, deg):
        (rows, cols) = shape
        a = int(cols / 2)
        b = a * deg

        centre = cols / 2.
        diff_centre = lambda g: g - centre if g > centre else centre - g

        new_kps = []
        for i in range(len(kps)):
            y, x = kps[i]
            add_height = int(((1. - (1. * diff_centre(y) / a) ** 2) * (b ** 2)) ** 0.5)
            height_ratio = 1 + (add_height * 2. / rows)

            min_row_in_consider = b - add_height
            new_x = min_row_in_consider + int(x * height_ratio)

            new_kps.append([y, new_x])

        return new_kps

    def spherical_calibration(self, shape, kps, fcl):
        rows, cols = shape
        x_mid = int(cols / 2)
        y_mid = int(rows / 2)
        s = fcl

        new_kps = []
        for i in range(len(kps)):
            x, y = kps[i]

            y_mirror = y - y_mid
            x_mirror = x - x_mid

            x_ = int(s * math.atan2(x_mirror, fcl))
            y_ = int(s * math.atan2(y_mirror, math.sqrt(x_mirror * x_mirror + fcl * fcl)))

            new_kps.append([x_ + x_mid, y_ + y_mid])

        return new_kps

    def findBound(self, img):
        rows, cols = img.shape[:2]

        zeros = np.zeros(cols, dtype='uint8')

        # print np.sum(img[0, :, 0]), len(img[0, :, 0])
        for i in range(rows):
            if np.array_equal(img[i, :, 0], zeros) == False:
                upper_rows = i
                break

        for i in range(rows - 1, -1, -1):
            if np.array_equal(img[i, :, 0], zeros) == False:
                lower_rows = i
                break
        return (upper_rows, lower_rows)

    def paste(self, left_img, right_img, moved):
        fade_min = int(moved)  # rightmost of left img will fade gradually
        fade_length = left_img.shape[1] - fade_min

        for j in range(left_img.shape[1]):

            if j >= fade_min:
                a_value = (1. * (left_img.shape[1] - j) / fade_length)
            else:
                a_value = 1
            for i in range (left_img.shape[0]):
                if left_img[i, j, 3] > 0:
                    right_img[i, j, :3] = left_img[i, j, :3] * a_value + (right_img[i, j, :3] * (1 - a_value)).astype(int)
                    right_img[i, j, 3] = 255

        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 2, 2, 2, 1],
                           [1, 2, 5, 2, 1],
                           [1, 2, 2, 2, 1],
                           [1, 1, 1, 1, 1]]) / 37.

        right_img[:, (fade_min):(fade_min + 3), :3] = cv2.filter2D(src=right_img[:, (fade_min):(fade_min + 3), :3],
                                                                    ddepth=-1, kernel=kernel)
        # right_img[:, (fade_min - 10):(fade_min + 10), :3] = 255


        return right_img

    def paste_4(self, left_img, right_img, moved):
        result = np.copy(right_img)

        blend_min = int(moved)
        blend_max = left_img.shape[1]
        blend_length = float(blend_max - blend_min)

        def diff_to_mid(x): return (blend_max - x) / blend_length

        result[:, :blend_min + 1, :] = left_img[:, :blend_min + 1, :]

        for j in range(blend_min, blend_max):
            p = diff_to_mid(j)

            for i in range(left_img.shape[0]):
                if np.array_equal(left_img[i, j, :], [0, 0, 0, 0]):
                    result[i, j, :] = right_img[i, j, :]
                elif np.array_equal(right_img[i, j, :], [0, 0, 0, 0]):
                    result[i, j, :] = left_img[i, j, :]
                else:
                    result[i, j, :] = (left_img[i, j, :] * p + right_img[i, j, :] * (1 - p)).astype('uint8')

        return result

    def paste_2(self, left_img, right_img, moved):

        centre_left = left_img.shape[1] / 2
        centre_right = (right_img.shape[1] + moved) / 2
        real_cols_right = (right_img.shape[1] - moved)
        real_cols_left = left_img.shape[1]

        rows, cols = right_img.shape[:2]

        getDiffLeff = lambda g: 1. * (g - centre_left) / real_cols_left if g > centre_left else 1. * (centre_left - g) / real_cols_left
        getDiffRight = lambda g: 1. * (g - centre_right) / real_cols_right if g > centre_right else 1. * (centre_right - g) / real_cols_right

        for i in range(rows):
            for j in range(cols):
                if j >= real_cols_left or left_img[i, j, 0] == 0:
                    pass
                elif right_img[i, j, 0] == 0:
                    right_img[i, j, :] = left_img[i, j, :]

                else:
                    prop_l = 1. - getDiffLeff(j)
                    prop_r = 1. - getDiffRight(j)
                    sum_diff = prop_l + prop_r
                    right_img[i, j, :] = (left_img[i, j, :] * prop_l / sum_diff + right_img[i, j, :] * prop_r / sum_diff).astype(int)

        # right_img[:, :, 3] = 255
        return right_img

    def paste_3(self, left_img, right_img, moved):
        fade_min = int(moved)  # rightmost of left img will fade gradually
        fade_length = left_img.shape[1] - fade_min

        right_img[:, :fade_min, :] = left_img[:, :fade_min, :]

        for j in range(fade_min, left_img.shape[1]):
            for i in range(left_img.shape[0]):
                if right_img[i, j, 0] != 0 and left_img[i, j, 0] != 0:
                    right_img[i, j, :3] = (left_img[i, j, :3] + right_img[i, j, 3]) / 2
                    right_img[i, j, 3] = 255

        return right_img

    def trimSurplusCols(self, img):

        rows, cols = img.shape[:2]

        for j in range(cols):
            meaningful = np.sum(img[:, j, 0] != 0)

            if  meaningful > 0:
                left_from = j
                break

        for j in range(cols - 1, -1, -1):
            meaningful = np.sum(img[:, j, 0] != 0)

            if  meaningful > 0:
                right_to = j
                break

        return img[:, left_from:right_to + 1, :]

    def detectAndDescribe(self, image):
        # print image.shape
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):

        matches = [(v, u) for u, v in matches]

        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 4), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

    def drawKps(self, input_image, kps):
        image = np.copy(input_image)
        # print 'Drawing'
        for kp in kps:
            cv2.circle(image, (int(kp[0]), int(kp[1])), 5, (0, 255, 0))
            # cv2.imshow('zzz', image)
            # cv2.waitKey()

        return image


def getPanorama(_images):
    images = []
    for _image in _images:
        images.append(load_image(_image))

    stitcher = Stitcher()
    result = stitcher.stitch([images[0], images[1]])

    cv2.imwrite('pano.jpg', result)

    return 'pano.jpg'


if __name__ == '__main__':

    begin_time = datetime.datetime.now()

    _imageDirectory = '/Users/tungphung/Documents/images15/'
    _imageList = [f for f in listdir(_imageDirectory)]
    _images = [join(_imageDirectory, f) for f in _imageList \
               if isfile(join(_imageDirectory, f)) and not f.startswith('.')]

    if len(_images) < 2:
        print '[>>Error<<] There is %d image' % (len(_images))
        exit(0)
    print '\n\nImage links: ', _images, '\n\n'

    imageA = load_image(_images[0])
    imageB = load_image(_images[1])
    imageC = load_image(_images[2])
    imageD = load_image(_images[3])

    stitcher = Stitcher()
    imageAB = stitcher.stitch([imageA, imageB])
    cv2.imshow('Pano AB', imageAB)

    stitcher = Stitcher()
    imageCD = stitcher.stitch([imageC, imageD])
    cv2.imshow('Pano CD', imageCD)

    stitcher = Stitcher()
    imageABCD = stitcher.stitch([imageAB, imageCD], calibrated=True)
    cv2.imshow('Pano ABCD', imageABCD)

    print 'elapsed time:', datetime.datetime.now() - begin_time

    cv2.imwrite('panoramo.jpg', imageABCD)
    cv2.waitKey(0)
