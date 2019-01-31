import cv2 as cv


class StereoParams:
    def __init__(self, intrinsic_filepath, extrinsic_filepath, image_size=None):
        if image_size is None:
            self._image_size = (752, 480)
        else:
            self._image_size = (image_size[0], image_size[1])

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

        r = cv.stereoRectify(self._M1, self._D1, self._M2, self._D2, self._image_size,
                             self._R, self._T, flags=cv.CALIB_ZERO_DISPARITY)
        self._R1, self._R2, self._P1, self._P2, _, _, _ = r

        self._map11, self._map12 = cv.initUndistortRectifyMap(
            self._M1, self._D1, self._R1, self._P1, self._image_size, cv.CV_16SC2)
        self._map21, self._map22 = cv.initUndistortRectifyMap(
            self._M2, self._D2, self._R2, self._P2, self._image_size, cv.CV_16SC2)

    def _remap(self, image, map1, map2):
        resized = False
        orig_size = (image.shape[1], image.shape[0])
        if orig_size != self._image_size:
            image = cv.resize(image, self._image_size)
            resized = True

        image = cv.remap(image, map1, map2, cv.INTER_LINEAR)

        if resized:
            image = cv.resize(image, orig_size)

        return image

    def remap_left(self, image):
        return self._remap(image, self._map11, self._map12)

    def remap_right(self, image):
        return self._remap(image, self._map21, self._map22)
