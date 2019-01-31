import cv2 as cv


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

        # TODO : init matchers and etc.
        # http://timosam.com/python_opencv_depthimage
        # https://github.com/hothanh/OpenCV_Depth/blob/master/stereo_depth_v2.cpp

    def __call__(self, left_frame, right_frame):
        disp = self._sgbm.compute(left_frame, right_frame)
        # TODO : apply matchers and filter
        return disp
