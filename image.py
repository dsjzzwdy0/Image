#-*- coding:utf-8 -*-

import os
import sys
from utils.point_feature import *
from utils.feature_matching import *
import cv2

path = os.path.dirname(os.path.realpath(sys.argv[0]))


def main():
    # max_point = 1024
    # extractor = OrbExtractor()
    # extractor = SiftExtractor(n_sample=1024)
    extractor = DeepExtractor(os.path.join(path, 'data/geodesc.pb'))
    extractor.create()
    matcher = MatcherWrapper()

    feature_matcher = FeatureMatcher(extractor, matcher)

    image1 = cv2.imread('images/tu1.jpg')
    image2 = cv2.imread('images/tu2.jpg')

    image_result = feature_matcher.compute_match_image(image1, image2)
    cv2.imshow('display', image_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()