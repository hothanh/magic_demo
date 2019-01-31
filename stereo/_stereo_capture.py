import cv2 as cv


class StereoCapture:
    def __init__(self, source, stereo_params=None):
        self._cap = cv.VideoCapture(source)
        # Needs a patch
        #self._cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'Y16 '))

        if stereo_params is None:
            self._stereo_params = None
        else:
            self._stereo_params = stereo_params

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, val):
        self._cap.set(prop, val)

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            return ret, (frame, frame)
        else:
            _, left, right = cv.split(frame)

            if self._stereo_params is None:
                return ret, (left, right)
            else:
                left = self._stereo_params.remap_left(left)
                right = self._stereo_params.remap_right(right)

                return ret, (left, right)

    def release(self):
        self._cap.release()
