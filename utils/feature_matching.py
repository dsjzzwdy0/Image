#!/usr/bin/env python
"""
Copyright 2019, Shijun Deng, Tigis.
Conduct pair-wise image matching.
"""
import time
import cv2


def draw_match(matcher_wrapper, img1, img2, feat1, feat2, kpts1, kpts2,
               dist_type, ratio, ransac=True, cross_check=True):
    good_matches, deep_mask = matcher_wrapper.get_matches(
        feat1, feat2, kpts1, kpts2, dist_type,
        ratio, ransac, cross_check, info='deep')

    display = matcher_wrapper.draw_matches(
        img1, kpts1, img2, kpts2, good_matches, deep_mask)
    return display, len(good_matches)


class FeatureMatcher:
    def __init__(self, extractor, matcher):
        self.extractor = extractor
        self.matcher = matcher

    def extract_image_feature(self, image):
        start = time.time()
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img1, gray_img, cv_kpts1, feat1 = self.extractor.get_image_keypnts(image, gray_img)
        end = time.time()
        print('Extract ', len(cv_kpts1), ' key points and features, total spend time is ', round(end - start, 3), 's.')

        return img1, gray_img, cv_kpts1, feat1

    def compute_match_image(self, image1, image2, dist_type=cv2.NORM_L2, ratio=0.70, cross_check=True, ransac=True):
        img1, gray_img1, cv_kpts1, feat1 = self.extract_image_feature(image1)
        img2, gray_img2, cv_kpts2, feat2 = self.extract_image_feature(image2)

        start = time.time()
        display, match_num = draw_match(self.matcher, img1, img2, feat1, feat2, cv_kpts1, cv_kpts2, dist_type,
                                  ratio, ransac, cross_check)
        end = time.time()
        print('Compute match points is',  match_num, ' total spend time is ', round(end - start, 3), 's.')

        return display


