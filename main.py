import config
import cv2 as cv
import numpy as np
import os
from detectors import SkeletonDetector, BODY_MODEL
from stereo import DisparityCalculator, StereoCapture, StereoParams, prepare_for_vis
import time

ROOT_DIR = os.path.dirname(__file__)


colors = [
    (11, 138, 236),
    (136, 7, 105),
    (200, 4, 4),
    (200, 4, 200),
    (242, 113, 211),
    (4, 4, 200),
    (4, 200, 200),
    (4, 200, 4),
]


def draw_skeleton(image, people_joints):
    rendered = image.copy()
    if rendered.ndim == 2 or rendered.shape[-1] == 1:
        rendered = cv.cvtColor(rendered, cv.COLOR_GRAY2BGR)

    for n, person in enumerate(people_joints):
        color = colors[n % len(colors)][::-1]
        for i in range(len(person)):
            if person[i] is not None:
                for j in BODY_MODEL[i][1]:
                    if person[j] is not None:
                        cv.line(rendered, (int(person[i][0]), int(person[i][1])),
                                (int(person[j][0]), int(person[j][1])), color, 1)

        joints = [p for p in person if p is not None]
        joints = np.reshape(joints, [len(joints), 1, 2])
        joints = np.int32(joints)
        cv.polylines(rendered, joints, True, (0, 0, 255), 2)

    return rendered


def main():
    skeleton_model_path = os.path.join(ROOT_DIR, 'models', 'pose-unet-64x96.pb')
    if not os.path.exists(skeleton_model_path):
        raise RuntimeError('Can\'t find a skeleton detector model!')

    intrinsics_path = os.path.join(ROOT_DIR, 'models', 'intrinsics.yml')
    if not os.path.exists(intrinsics_path):
        raise RuntimeError('Can\'t find a intrinsics file!')

    extrinsics_path = os.path.join(ROOT_DIR, 'models', 'extrinsics.yml')
    if not os.path.exists(extrinsics_path):
        raise RuntimeError('Can\'t find a extrinsics file!')

    skeleton_detector = SkeletonDetector(skeleton_model_path, (96, 64), 6)

    stereo_params = StereoParams(intrinsics_path, extrinsics_path)

    disp_calc = DisparityCalculator(**config.SGBM_params)
    cap = StereoCapture(0, stereo_params)

    print('Capture: (%i, %i) %.2f' % (int(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), cap.get(cv.CAP_PROP_FPS)), flush=True)

    if config.DEBUG_WRITE:
        writer = 0
        disp_writer = None
        left_writer = None
        rigt_writer = None

    try:
        print('Processing is started.', flush=True)

        start = time.time()
        while True:
            cap_start = time.time()
            ret, (left_frame, right_frame) = cap.read()
            cap_elapsed = time.time() - cap_start
            if not ret:
                break

            disp_start = time.time()
            disparity_map = disp_calc(cv.pyrDown(left_frame), cv.pyrDown(right_frame))
            disparity_map = cv.pyrUp(disparity_map)
            disp_elapsed = time.time() - disp_start

            skeleton_start = time.time()
            people = skeleton_detector(left_frame, prepare_for_vis(disparity_map))
            skeleton_elapsed = time.time() - skeleton_start

            people_bboxes = []
            for person in people:
                head = person[0]
                neck = person[1]
                if head is not None and neck is not None:
                    dist = np.linalg.norm(np.array(head)-neck)
                    bbox = [head[0]-dist, head[1]-dist, head[0]+dist, head[1]+dist]
                else:
                    bbox = None
                people_bboxes.append(bbox)

            end = time.time()
            elapsed = end - start
            frame_elapsed = end - cap_start
            start = end

            print('\rFPS: %.2f; CAP: %.2f; DISP: %.2f; SKLTN: %.2f; FRAME: %.2f' %
                  (1/elapsed, 1/cap_elapsed, 1/disp_elapsed, 1/skeleton_elapsed, 1/frame_elapsed), end='', flush=True)

            display_image = draw_skeleton(left_frame, people)

            cv.imshow('demo', display_image)

            if config.DEBUG_WRITE:
                if isinstance(writer, int):
                    writer += 1
                    if writer > 2:
                        writer = cv.VideoWriter('./demo.mp4', cv.VideoWriter_fourcc(*'avc1'), 1/elapsed, (int(
                            cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
                        disp_writer = cv.VideoWriter('./demo-disps.mp4', cv.VideoWriter_fourcc(*'avc1'), 1/elapsed, (int(
                            cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), False)
                        left_writer = cv.VideoWriter('./demo-lefts.mp4', cv.VideoWriter_fourcc(*'avc1'), 1/elapsed, (int(
                            cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), False)
                        right_writer = cv.VideoWriter('./demo-rights.mp4', cv.VideoWriter_fourcc(*'avc1'), 1/elapsed, (int(
                            cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), False)
                else:
                    writer.write(display_image)
                    disp_writer.write(prepare_for_vis(disparity_map))
                    left_writer.write(left_frame)
                    right_writer.write(right_frame)

            key = cv.waitKey(1)
            if key == 27:
                break
            if key & 0xFF == ord('q'):
                break
    except:
        print('', flush=True)
        raise
    finally:
        cap.release()
        if config.DEBUG_WRITE:
            if not isinstance(writer, int):
                writer.release()
                disp_writer.release()
                left_writer.release()
                right_writer.release()


if __name__ == "__main__":
    main()
