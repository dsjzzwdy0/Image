#!/usr/bin/env python
"""
Copyright 2019, Shijun Deng, Tigis.
Conduct pair-wise image matching.
"""
import cv2


def draw_match(matcher_wrapper, img1, img2, feat1, feat2, kpts1, kpts2,
               dist_type, ratio, ransac=True, cross_check=True):
    deep_good_matches, deep_mask = matcher_wrapper.get_matches(
        feat1, feat2, kpts1, kpts2, dist_type,
        ratio, ransac, cross_check, info='deep')

    deep_display = matcher_wrapper.draw_matches(
        img1, kpts1, img2, kpts2, deep_good_matches, deep_mask)
    return deep_display


class FeatureMatcher:
    def __init__(self, extractor, matcher):
        self.extractor = extractor
        self.matcher = matcher

    def match(self, image1, image2, dist_type=cv2.NORM_L2, ratio=0.70, cross_check=True, ransac=True):
        gray_img1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray_img2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        img1, gray_img1, cv_kpts1, feat1 = self.extractor.get_image_keypnts(image1, gray_img1)
        img2, gray_img2, cv_kpts2, feat2 = self.extractor.get_image_keypnts(image2, gray_img2)

        sift_display = draw_match(self.matcher, img1, img2, feat1, feat2,
                                  cv_kpts1, cv_kpts2, dist_type,
                                  ratio, ransac, cross_check)
        return sift_display


