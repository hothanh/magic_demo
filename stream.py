import config
import cv2 as cv
import numpy as np
import os
from detectors import SkeletonDetector, BODY_MODEL
from stereo import DisparityCalculator, StereoCapture, StereoParams, prepare_for_vis
import time
import random

ROOT_DIR = os.path.dirname(__file__)

def main():
    intrinsics_path = os.path.join(ROOT_DIR, 'models', 'intrinsics.yml')
    if not os.path.exists(intrinsics_path):
        raise RuntimeError('Can\'t find a intrinsics file!')

    extrinsics_path = os.path.join(ROOT_DIR, 'models', 'extrinsics.yml')
    if not os.path.exists(extrinsics_path):
        raise RuntimeError('Can\'t find a extrinsics file!')

    skeleton_model_path = os.path.join(ROOT_DIR, 'models', 'pose-unet-128x160.pb')
    if not os.path.exists(skeleton_model_path):
        raise RuntimeError('Can\'t find a skeleton detector model!')
    stereo_params = StereoParams(intrinsics_path, extrinsics_path)
    if isinstance(config.VIDEO_SOURCE, int):
        cap = StereoCapture(config.VIDEO_SOURCE, stereo_params)
    else:
        cap = StereoCapture(config.VIDEO_SOURCE)
    while True:
        ret, (left_frame, right_frame) = cap.read()
        cv.imshow('left',left_frame);
        cv.imshow('right',right_frame);
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
