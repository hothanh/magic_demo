import cv2 as cv


class StereoCapture:
    def __init__(self, source, intrinsic_filepath=None, extrinsic_filepath=None):
        self._cap = cv.VideoCapture(source)
        self._cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'y16 '))

        self._cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        self._image_size = (
            752,  # int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            480  # , int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        )

        if intrinsic_filepath is None and extrinsic_filepath is None:
            self._calibrated = False
        if intrinsic_filepath is not None and extrinsic_filepath is not None:
            intrinsic_file = cv.FileStorage(intrinsic_filepath, cv.FileStorage_READ)
            if not intrinsic_file.isOpened():
                raise ValueError('Cannot open intrinsic file')

            extrinsic_file = cv.FileStorage(extrinsic_filepath, cv.FileStorage_READ)
            if not extrinsic_file.isOpened():
                raise ValueError('Cannot open extrinsic file')

            self._M1 = intrinsic_file.getNode('M1').mat()
            self._D1 = intrinsic_file.getNode('D1').mat()
            self._M2 = intrinsic_file.getNode('M2').mat()
            self._D2 = intrinsic_file.getNode('D2').mat()

            self._R = extrinsic_file.getNode('R').mat()
            self._T = extrinsic_file.getNode('T').mat()

            self._compute_rectify_params()

            self._calibrated = True
        else:
            raise ValueError('Both intrinsic and extrinsic files must be set.')

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            return ret, (frame, frame)
        else:
            _, left, right = cv.split(frame)

            if not self._calibrated:
                return ret, (left, right)
            else:
                left_resized = False
                left_orig_size = (left.shape[1], left.shape[0])
                if left_orig_size != self._image_size:
                    left = cv.resize(left, self._image_size)
                    left_resized = True

                right_resized = False
                right_orig_size = (right.shape[1], right.shape[0])
                if right_orig_size != self._image_size:
                    right = cv.resize(right, self._image_size)
                    right_resized = True

                left = cv.remap(left, self._map11, self._map12, cv.INTER_LINEAR)
                right = cv.remap(right, self._map21, self._map22, cv.INTER_LINEAR)

                if left_resized:
                    left = cv.resize(left, left_orig_size)

                if right_resized:
                    right = cv.resize(right, right_orig_size)

                return ret, (left, right)

    def _compute_rectify_params(self):
        r = cv.stereoRectify(self._M1, self._D1, self._M2, self._D2, self._image_size,
                             self._R, self._T, flags=cv.CALIB_ZERO_DISPARITY)
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = r

        self._map11, self._map12 = cv.initUndistortRectifyMap(self._M1, self._D1, R1, P1, self._image_size, cv.CV_16SC2)
        self._map21, self._map22 = cv.initUndistortRectifyMap(self._M2, self._D2, R2, P2, self._image_size, cv.CV_16SC2)
