import cv2 as cv
import numpy as np


def prepare_for_vis(disp):
    disp = disp.copy()
    cv.normalize(disp, dst=disp, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
    return np.uint8(disp)


class DisparityCalculator:
    def __init__(self, **kwargs):
        window_size = kwargs.get('window_size', 3)
        self._sgbm = cv.StereoSGBM_create()
        self._sgbm.setPreFilterCap(kwargs.get('pre_filter_cap', 0))
        self._sgbm.setBlockSize(max(3, kwargs.get('block_size', 13)))
        self._sgbm.setP1(kwargs.get('P1', 8 * 3 * window_size ** 2))
        self._sgbm.setP2(kwargs.get('P2', 32 * 3 * window_size ** 2))
        self._sgbm.setMinDisparity(kwargs.get('mininum_disparity', 0))
        self._sgbm.setNumDisparities(kwargs.get('number_of_disparities', 160))
        self._sgbm.setUniquenessRatio(kwargs.get('uniqe_ratio', 15))
        self._sgbm.setSpeckleWindowSize(kwargs.get('speckle_win_size', 400))
        self._sgbm.setSpeckleRange(kwargs.get('speckle_range', 200))
        self._sgbm.setDisp12MaxDiff(kwargs.get('disp12_max_diff', 1))
        self._sgbm.setMode(kwargs.get('mode', 1))

        if not hasattr(cv, 'ximgproc'):
            self._right_matcher = None
        else:
            self._right_matcher = cv.ximgproc.createRightMatcher(self._sgbm)

            self._wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=self._sgbm)
            self._wls_filter.setLambda(kwargs.get('wls_lambda', 8000.0))
            self._wls_filter.setSigmaColor(kwargs.get('wls_sigma', 1.5))

    def __call__(self, left_frame, right_frame):
        left_disp = self._sgbm.compute(left_frame, right_frame)
        if self._right_matcher is None:
            return left_disp
        else:
            right_disp = self._right_matcher.compute(right_frame, left_frame)

            left_disp = np.int16(left_disp)
            right_disp = np.int16(right_disp)

            disp = self._wls_filter.filter(left_disp, left_frame, None, right_disp)

            return disp
