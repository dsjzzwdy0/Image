#-*- coding:utf-8 -*-

import os
import sys
from utils.point_feature import *
from utils.feature_matching import *
import cv2

path = os.path.dirname(os.path.realpath(sys.argv[0]))


def getAdaptRate(image_width, image_height):
    width = 1600
    height = 860
    r0 = float(width) / image_width
    r1 = float(height) / image_height

    r = r0 if r1 > r0 else r1
    return r


def main():
    # max_point = 1024
    # extractor = OrbExtractor()
    # extractor = SiftExtractor(n_sample=1024)
    extractor = DeepExtractor(os.path.join(path, 'data/geodesc.pb'))
    extractor.create()
    matcher = MatcherWrapper()

    feature_matcher = FeatureMatcher(extractor, matcher)

    image1 = cv2.imread('D:/Python/images/qlh036-1.jpg')
    image2 = cv2.imread('D:/Python/images/qlh037-1.jpg')

    image_result = feature_matcher.compute_match_image(image1, image2)

    shape = image_result.shape
    r = getAdaptRate(shape[1], shape[0])

    img_test1 = cv2.resize(image_result, (int(shape[1] * r), int(shape[0] * r)))

    cv2.imshow('display', img_test1)
    cv2.waitKey(0)
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()