import cv2 as cv
import numpy as np
import xunit


class _StereoCaptureFileImpl:
    def __init__(self, left_video, right_video):
        self._left_cap = cv.VideoCapture(left_video)
        self._right_cap = cv.VideoCapture(right_video)

        if not self._left_cap.isOpened():
            raise ValueError('Can\'t open a video for a left channles')
        if not self._right_cap.isOpened():
            raise ValueError('Can\'t open a video for a left channles')

    def get(self, prop):
        return self._left_cap.get(prop)

    def set(self, prop, val):
        self._left_cap.set(prop, val)
        self._right_cap.set(prop, val)

    def read(self):
        ret1, left_frame = self._left_cap.read()
        ret2, right_frame = self._right_cap.read()
        if not (ret1 & ret2):
            return False, left_frame
        else:
            if left_frame.ndim > 2:
                if left_frame.shape[-1] == 1:
                    left_frame = left_frame[..., 0]
                else:
                    left_frame = cv.cvtColor(left_frame, cv.COLOR_BGR2GRAY)

            if right_frame.ndim > 2:
                if right_frame.shape[-1] == 1:
                    right_frame = right_frame[..., 0]
                else:
                    right_frame = cv.cvtColor(right_frame, cv.COLOR_BGR2GRAY)

            frame = cv.merge([np.zeros_like(left_frame), left_frame, right_frame])
            return True, frame

    def release(self):
        self._left_cap.release()
        self._right_cap.release()


class StereoCapture:
    def __init__(self, source, stereo_params=None):
        self._mono = False
        if isinstance(source, int):
            self._cap = cv.VideoCapture(source)
            #self._cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'Y16 '))
        elif isinstance(source, tuple) and len(source) == 2:
            if isinstance(source[0], str) and isinstance(source[1], str):
                self._cap = _StereoCaptureFileImpl(*source)
            else:
                self._cap = cv.VideoCapture(source[0])
                self._mono = bool(source[1])

        if stereo_params is None:
            self._stereo_params = None
        else:
            self._stereo_params = stereo_params
        if not xunit.InitExtensionUnit('usb-c0030000.ehci-1.1'):
            print('InitExtensionUnit failed')
        if not xunit.SetAutoExposureStereo():
            print('SetAutoExposureStereo failed')

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, val):
        self._cap.set(prop, val)

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            return ret, (frame, frame)
        else:
            if self._mono:
                second = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                first = second
            else:
                _, second, first = cv.split(frame)

            if self._stereo_params is None:
                return ret, (first, second)
            else:
                left = self._stereo_params.remap_left(first)
                right = self._stereo_params.remap_right(second)

                return ret, (left, right)

    def release(self):
        self._cap.release()
