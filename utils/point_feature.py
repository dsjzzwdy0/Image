#!/usr/bin/env python
"""
Copyright 2019, Shijun Deng, Tigis.
Conduct pair-wise image matching.
"""

import cv2
import os
import numpy as np
import time
import queue
import tensorflow as tf
from threading import Thread


def load_frozen_model(pb_path, prefix='', print_nodes=False):
    """Load frozen model (.pb file) for testing.
    After restoring the model, operators can be accessed by
    graph.get_tensor_by_name('<prefix>/<op_name>')
    Args:
        pb_path: the path of frozen model.
        prefix: prefix added to the operator name.
        print_nodes: whether to print node names.
    Returns:
        graph: tensorflow graph definition.
    """
    if os.path.exists(pb_path):
        with tf.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                name=prefix
            )
            if print_nodes:
                for op in graph.get_operations():
                    print(op.name)
            return graph
    else:
        print('Model file does not exist', pb_path)
        exit(-1)


class FeatureExtractor(object):
    def __init__(self):
        pass

    def create(self):
        pass

    def detect(self, gray_img):
        kpts = None
        desc = None
        return kpts, desc

    def get_image_keypnts(self, img, gray_img):
        kpts = None
        desc = None
        return img, gray_img, kpts, desc


class MatcherWrapper(object):
    """
    OpenCV matcher wrapper.
    """
    def __init__(self):
        pass

    def get_matches(self, feat1, feat2, cv_kpts1, cv_kpts2, dist_type, ransac=True,
                    ratio=None, cross_check=True, info=''):
        """
        Compute putative and inlier matches.
        Args:
            feat: (n_kpts, 128) Local features.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
            dist_type: Dist type parameter, etc: cv2.NORM_L2
            ratio: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            info: Info to print out.
        Returns:
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """
        matcher = cv2.BFMatcher(dist_type)    #cv2.NORM_L2
        good_matches = []
        mask = None

        start = time.time()
        if(cross_check):
            init_matches1 = matcher.knnMatch(feat1, feat2, k=2)
            init_matches2 = matcher.knnMatch(feat2, feat1, k=2)
            for i in range(len(init_matches1)):
                # cross check
                if cross_check and init_matches2[init_matches1[i][0].trainIdx][0].trainIdx == i:
                    # ratio test
                    if ratio is not None and init_matches1[i][0].distance <= ratio * init_matches1[i][1].distance:
                        good_matches.append(init_matches1[i][0])
                    elif ratio is None:
                        good_matches.append(init_matches1[i][0])
                elif not cross_check:
                    good_matches.append(init_matches1[i][0])
        else:
            raw_matches = matcher.knnMatch(feat1, feat2, k=2)
            for m, n in raw_matches:
                if m.distance < ratio * n.distance:
                    good_matches.append(m)

        if(ransac):
            good_kpts1 = np.array([cv_kpts1[m.queryIdx].pt for m in good_matches])
            good_kpts2 = np.array([cv_kpts2[m.trainIdx].pt for m in good_matches])
            _, mask = cv2.findFundamentalMat(good_kpts1, good_kpts2, cv2.RANSAC, 4.0, confidence=0.999)

        end = time.time()
        print('Time cost in feature match ', end - start)

        n_inlier = np.count_nonzero(mask)
        print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)

        return good_matches, mask

    def draw_matches(self, img1, cv_kpts1, img2, cv_kpts2, good_matches, mask,
                     match_color=(0, 255, 0), pt_color=(0, 0, 255), flags=2):
        """Draw matches."""
        display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches,
                                  None,
                                  matchColor=match_color,
                                  singlePointColor=pt_color,
                                  matchesMask=None if mask is None else mask.ravel().tolist(),
                                  flags=flags)  #=4
        return display


class DeepExtractor(FeatureExtractor):
    def __init__(self, model_path, n_sample=1024, batch_size=512):
        self.n_sample = n_sample
        self.sift_extractor = None
        self.batch_size = batch_size
        self.model_path = model_path
        self.sess = None
        self.graph = None

    def create(self):
        self.sift_extractor = SiftExtractor(n_sample=self.n_sample)
        self.sift_extractor.create()
        self.graph = load_frozen_model(self.model_path, print_nodes=False)


    def detect(self, gray_img):
        print("SIFT Extractor is Null = ", self.sift_extractor is None)
        _, cv_kpts1 = self.sift_extractor.detect(gray_img)

        self.sess = tf.Session(graph=self.graph)
        # extract deep feature from images.
        deep_feat1, cv_kpts1, gray_img1 = self.sift_extractor.extract_deep_features(
            self.sess, gray_img, cv_kpts1, self.batch_size, qtz=True)

        print("Img1 key point size ", len(cv_kpts1), ", feature size ", len((deep_feat1)))
        self.sess.close()

        return cv_kpts1, deep_feat1

    def get_image_keypnts(self, img, gray_img):
        # detect SIFT keypoints.
        start = time.time()
        cv_kpts, des = self.detect(gray_img)
        end = time.time()
        print('Time cost in keypoint detection', end - start)
        return img, gray_img, cv_kpts, des


class FastExtractor(FeatureExtractor):
    '''
    FAST特征提取类
    '''
    def __init__(self, threshold=40, nonmaxSuppression=True,
                 type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16):
        self.threshold = threshold
        self.nonmaxSuppression = nonmaxSuppression
        self.type = type
        self.fast = None
        self.brief = None

    def create(self):
        self.fast = cv2.FastFeatureDetector_create(self.threshold, self.nonmaxSuppression, self.type)
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    def detect(self, gray_Img):
        kpts = self.fast.detect(gray_Img, None)
        kp, des = self.brief.compute(gray_Img, kpts)
        return kp, des

    def get_image_keypnts(self, img, gray_img):
        # detect SIFT keypoints.
        start = time.time()
        cv_kpts, des = self.detect(gray_img)
        end = time.time()
        print('Time cost in keypoint detection', end - start)
        return img, gray_img, cv_kpts, des


class OrbExtractor(FeatureExtractor):
    '''
    ORB特征提取类
    '''
    def __init__(self,
                 nfeatures=800,
                 scaleFactor=1.2,
                 nlevels=8,
                 edgeThreshold=31,
                 firstLevel=0,
                 WTA_K=2,
                 scoreType=cv2.ORB_HARRIS_SCORE,
                 patchSize=31,
                 fastThreshold=20):
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        self.edgeThreshold = edgeThreshold
        self.firstLevel = firstLevel
        self.WTA_K = WTA_K
        self.scoreType = scoreType
        self.patchSize = patchSize
        self.fastThreshold = fastThreshold
        self.orb = None

    def create(self):
        self.orb = cv2.ORB_create(self.nfeatures, self.scaleFactor, self.nlevels,
                                  self.edgeThreshold, self.firstLevel, self.WTA_K,
                                  self.scoreType, self.patchSize, self.fastThreshold)

    def detect(self, gray_Img):
        kp, des = self.orb.detectAndCompute(gray_Img, None)
        return kp, des

    def get_image_keypnts(self, img, gray_img):
        # detect SIFT keypoints.
        start = time.time()
        cv_kpts, des = self.detect(gray_img)
        end = time.time()
        print('Time cost in keypoint detection', end - start)
        return img, gray_img, cv_kpts, des


class SurfExtractor(FeatureExtractor):
    '''
    SURF特征点提取类
    '''
    def __init__(self, hessian_threshold=1000):
        self.hessian_threshold = hessian_threshold
        self.surf = None

    def create(self):
        self.surf = cv2.xfeatures2d.SURF_create(self.hessian_threshold)

    def detect(self, gray_Img):
        kp, des = self.surf.detectAndCompute(gray_Img, None)
        return kp, des

    def get_image_keypnts(self, img, gray_img):
        # detect SIFT keypoints.
        start = time.time()
        cv_kpts, des = self.detect(gray_img)
        end = time.time()
        print('Time cost in keypoint detection', end - start)
        return img, gray_img, cv_kpts, des


class SiftExtractor(FeatureExtractor):
    """"
    OpenCV SIFT 特征提取类.
    """
    def __init__(self,
                 nfeatures=0, n_octave_layers=3,
                 peak_thld=0.0067, edge_thld=10, sigma=1.6,
                 n_sample=8192, patch_size=32):
        self.sift = None
        self.nfeatures = nfeatures
        self.n_octave_layers = n_octave_layers
        self.peak_thld = peak_thld
        self.edge_thld = edge_thld
        self.sigma = sigma
        self.n_sample = n_sample
        self.down_octave = True

        self.sift_init_sigma = 0.5
        self.sift_descr_scl_fctr = 3.
        self.sift_descr_width = 4

        self.first_octave = None
        self.max_octave = None
        self.pyr = None

        self.patch_size = patch_size
        self.output_gird = None

    def create(self):
        """Create OpenCV SIFT detector."""
        self.sift = cv2.xfeatures2d.SIFT_create(self.nfeatures, self.n_octave_layers,
                                                self.peak_thld, self.edge_thld, self.sigma)

    def get_image_keypnts(self, img, gray_img):
        # detect SIFT keypoints.
        start = time.time()
        _, cv_kpts = self.detect(gray_img)
        des = self.compute_desc(img, cv_kpts)
        end = time.time()
        print('Time cost in keypoint detection', end - start)
        return img, gray_img, cv_kpts, des

    def detect(self, gray_img):
        """Detect keypoints in the gray-scale image.
        Args:
            gray_img: The input gray-scale image.
        Returns:
            npy_kpts: (n_kpts, 6) Keypoints represented as NumPy array.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
        """
        cv_kpts = self.sift.detect(gray_img, None)
        all_octaves = [np.int8(i.octave & 0xFF) for i in cv_kpts]

        self.first_octave = int(np.min(all_octaves))
        self.max_octave = int(np.max(all_octaves))

        '''print("All octaves size is ", len(all_octaves),
              ", max_octave is ", self.max_octave,
              ", min_octave is", self.first_octave,
              "; list is", all_octaves)
        '''

        npy_kpts, cv_kpts = sample_by_octave(cv_kpts, self.n_sample, self.down_octave)
        return npy_kpts, cv_kpts

    def compute_desc(self, img, cv_kpts):
        """Compute SIFT descriptions on given keypoints.
        Args:
            img: The input image, can be either color or gray-scale.
            cv_kpts: A list of cv2.KeyPoint.
        Returns:
            sift_desc: (n_kpts, 128) SIFT descriptions.
        """

        _, sift_desc = self.sift.compute(img, cv_kpts)
        return sift_desc

    def build_pyramid(self, gray_img):
        """Build pyramid. It would be more efficient to use the pyramid
        constructed in the detection step.
        Args:
            gray_img: Input gray-scale image.
        Returns:
            pyr: A list of gaussian blurred images (gaussian scale space).
        """
        gray_img = gray_img.astype(np.float32)
        n_octaves = self.max_octave - self.first_octave + 1

        print("n_octaves number is ", n_octaves)

        # create initial image.
        if self.first_octave < 0:
            sig_diff = np.sqrt(np.maximum(
                np.square(self.sigma) - np.square(self.sift_init_sigma) * 4, 0.01))
            base = cv2.resize(gray_img, (gray_img.shape[1] * 2, gray_img.shape[0] * 2),
                              interpolation=cv2.INTER_LINEAR)
            base = cv2.GaussianBlur(base, None, sig_diff)
        else:
            sig_diff = np.sqrt(np.maximum(np.square(self.sigma) -
                                          np.square(self.sift_init_sigma), 0.01))
            base = cv2.GaussianBlur(gray_img, None, sig_diff)
        # compute gaussian kernels.
        sig = np.zeros((self.n_octave_layers + 3,))
        self.pyr = [None] * (n_octaves * (self.n_octave_layers + 3))
        sig[0] = self.sigma
        k = np.power(2, 1. / self.n_octave_layers)
        for i in range(1, self.n_octave_layers + 3):
            sig_prev = np.power(k, i - 1) * self.sigma
            sig_total = sig_prev * k
            sig[i] = np.sqrt(sig_total * sig_total - sig_prev * sig_prev)
        # construct gaussian scale space.
        for o in range(0, n_octaves):
            for i in range(0, self.n_octave_layers + 3):
                if o == 0 and i == 0:
                    dst = base
                elif i == 0:
                    src = self.pyr[(o - 1) * (self.n_octave_layers + 3) + self.n_octave_layers]
                    dst = cv2.resize(
                        src, (int(src.shape[1] / 2), int(src.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
                else:
                    src = self.pyr[o * (self.n_octave_layers + 3) + i - 1]
                    dst = cv2.GaussianBlur(src, None, sig[i])
                self.pyr[o * (self.n_octave_layers + 3) + i] = dst

    def get_patches(self, cv_kpts):
        """Get all patches around given keypoints.
        Args:
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
        Return:
            all_patches: (n_kpts, 32, 32) Cropped patches.
        """
        # generate sampling grids.
        n_pixel = np.square(self.patch_size)
        self.output_grid = np.zeros((n_pixel, 3), dtype=np.float32)
        for i in range(n_pixel):
            self.output_grid[i, 0] = (i % self.patch_size) * 1. / self.patch_size * 2 - 1
            self.output_grid[i, 1] = (i / self.patch_size) * 1. / self.patch_size * 2 - 1
            self.output_grid[i, 2] = 1

        scale_index = [[] for i in range(len(self.pyr))]
        for idx, val in enumerate(cv_kpts):
            octave, layer, _ = unpack_octave(val)
            scale_val = (int(octave) - self.first_octave) * (self.n_octave_layers + 3) #+ int(layer)
            if scale_val >= len(self.pyr):
                print("Index[", idx, "] The scale value is ", scale_val, ", but the pyr is", len(self.pyr))
                scale_val = len(self.pyr) - 1
            scale_index[scale_val].append(idx)

        all_patches = []
        for idx, val in enumerate(scale_index):
            tmp_cv_kpts = [cv_kpts[i] for i in val]
            scale_img = self.pyr[idx]
            # radius = self.sift_descr_scl_fctr * size * np.sqrt(2) * (self.sift_descr_width + 1) * 0.5
            patches = get_interest_region(scale_img, tmp_cv_kpts, self.output_grid,
                                          self.sift_descr_width, self.sift_descr_scl_fctr,
                                          self.patch_size)
            if patches is not None:
                all_patches.append(patches)

        if self.down_octave:
            all_patches = np.concatenate(all_patches[::-1], axis=0)
        else:
            all_patches = np.concatenate(all_patches, axis=0)
        assert len(cv_kpts) == all_patches.shape[0]
        return all_patches

    def extract_deep_features(self, sess, gray_img, cv_kpts, batch_size, qtz=True):
        start = time.time()
        self.build_pyramid(gray_img)
        end = time.time()
        print('Time cost in scale space construction', end - start)

        start = time.time()
        all_patches = self.get_patches(cv_kpts)
        end = time.time()
        print('Time cost in patch cropping', end - start)

        num_patch = all_patches.shape[0]

        if num_patch % batch_size > 0:
            loop_num = int(np.floor(float(num_patch) / float(batch_size)))
        else:
            loop_num = int(num_patch / batch_size - 1)

        def _worker(patch_queues, session, features):
            """The worker thread."""
            while True:
                patch_data = patch_queues.get()
                if patch_data is None:
                    return
                feat = session.run("squeeze_1:0", feed_dict={"input:0": np.expand_dims(patch_data, -1)})
                features.append(feat)
                patch_queues.task_done()

        all_feat = []
        patch_queue = queue.Queue()
        worker_thread = Thread(target=_worker, args=(patch_queue, sess, all_feat))
        worker_thread.daemon = True
        worker_thread.start()

        start = time.time()
        # enqueue
        for i in range(loop_num + 1):
            if i < loop_num:
                patch_queue.put(all_patches[i * batch_size: (i + 1) * batch_size])
            else:
                patch_queue.put(all_patches[i * batch_size:])
        # poison pill
        patch_queue.put(None)
        # wait for extraction.
        worker_thread.join()

        end = time.time()
        print('Time cost in feature extraction', end - start)

        all_feat = np.concatenate(all_feat, axis=0)
        # quantize output features.
        all_feat = (all_feat * 128 + 128).astype(np.uint8) if qtz else all_feat
        return all_feat, cv_kpts, gray_img


def unpack_octave(kpt):
    """Get scale coefficients of a keypoints.
    Args:
       kpt: A keypoint object represented as cv2.KeyPoint.
    Returns:
        octave: The octave index.
        layer: The level index.
        scale: The sampling step.
    """
    octave = kpt.octave & 255
    layer = (kpt.octave >> 8) & 255
    octave = octave if octave < 128 else (-128 | octave)
    scale = 1. / (1 << octave) if octave >= 0 else float(1 << -octave)
    return octave, layer, scale


def sample_by_octave(cv_kpts, n_sample, down_octave=True):
    """Sample keypoints by octave.
    Args:
        cv_kpts: The list of keypoints representd as cv2.KeyPoint.
        n_sample: The sampling number of keypoint. Leave to -1 if no sampling needed
        down_octave: (True by default) Perform sampling downside of octave.
    Returns:
        npy_kpts: (n_kpts, 5) Keypoints in NumPy format, represenetd as
                  (x, y, size, orientation, octave).
        cv_kpts: A list of sampled cv2.KeyPoint.
    """

    n_kpts = len(cv_kpts)
    npy_kpts = np.zeros((n_kpts, 5))
    for idx, val in enumerate(cv_kpts):
        npy_kpts[idx, 0] = val.pt[0]
        npy_kpts[idx, 1] = val.pt[1]
        npy_kpts[idx, 2] = val.size
        npy_kpts[idx, 3] = val.angle * np.pi / 180.
        npy_kpts[idx, 4] = np.int8(val.octave & 0xFF)

    if down_octave:
        sort_idx = (-npy_kpts[:, 2]).argsort()
    else:
        sort_idx = (npy_kpts[:, 2]).argsort()

    npy_kpts = npy_kpts[sort_idx]
    cv_kpts = [cv_kpts[i] for i in sort_idx]

    if n_sample > -1 and n_kpts > n_sample:
        # get the keypoint number in each octave.
        _, unique_counts = np.unique(npy_kpts[:, 4], return_counts=True)

        if down_octave:
            unique_counts = list(reversed(unique_counts))

        n_keep = 0
        for i in unique_counts:
            if n_keep < n_sample:
                n_keep += i
            else:
                break
        print('Sampled', n_keep, 'from', n_kpts)
        npy_kpts = npy_kpts[:n_keep]
        cv_kpts = cv_kpts[:n_keep]

    return npy_kpts, cv_kpts


def get_interest_region(scale_img, cv_kpts, output_grid, width, factor, patch_size, standardize=True):
    """Get the interest region around a keypoint.
    Args:
        scale_img: DoG image in the scale space.
        cv_kpts: A list of OpenCV keypoints.
        output_grid: the output value for the patch
        width: the width value for patch
        factor: the factor value for the width
        patch_size: the size of patch block
        standardize: (True by default) Whether to standardize patches as network inputs.
    Returns:
        Nothing.
    """
    batch_input_grid = []
    all_patches = []
    bs = 30  # limited by OpenCV remap implementation
    for idx, cv_kpt in enumerate(cv_kpts):
        # preprocess
        _, _, scale = unpack_octave(cv_kpt)
        size = cv_kpt.size * scale * 0.5
        ptf = (cv_kpt.pt[0] * scale, cv_kpt.pt[1] * scale)
        ori = (360. - cv_kpt.angle) * (np.pi / 180.)
        radius = np.round(factor * size * np.sqrt(2)* (width + 1) * 0.5)    #self.sift_descr_scl_fctr * size * np.sqrt(2)* (self.sift_descr_width + 1) * 0.5
        radius = np.minimum(radius, np.sqrt(np.sum(np.square(scale_img.shape))))
        # construct affine transformation matrix.
        affine_mat = np.zeros((3, 2), dtype=np.float32)
        m_cos = np.cos(ori) * radius
        m_sin = np.sin(ori) * radius
        affine_mat[0, 0] = m_cos
        affine_mat[1, 0] = m_sin
        affine_mat[2, 0] = ptf[0]
        affine_mat[0, 1] = -m_sin
        affine_mat[1, 1] = m_cos
        affine_mat[2, 1] = ptf[1]
        # get input grid.
        input_grid = np.matmul(output_grid, affine_mat)
        input_grid = np.reshape(input_grid, (-1, 1, 2))
        batch_input_grid.append(input_grid)

        if len(batch_input_grid) != 0 and len(batch_input_grid) % bs == 0 or idx == len(cv_kpts) - 1:
            # sample image pixels.
            batch_input_grid_ = np.concatenate(batch_input_grid, axis=0)
            patches = cv2.remap(scale_img.astype(np.float32), batch_input_grid_,
                                None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            patches = np.reshape(patches, (len(batch_input_grid),
                                           patch_size, patch_size))
            # standardize patches.
            if standardize:
                patches = (patches - np.mean(patches, axis=(1, 2), keepdims=True)) / \
                    (np.std(patches, axis=(1, 2), keepdims=True) + 1e-8)
            all_patches.append(patches)
            batch_input_grid = []

    if len(all_patches) != 0:
        all_patches = np.concatenate(all_patches, axis=0)
    else:
        all_patches = None
    return all_patches