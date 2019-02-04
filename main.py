import config
import cv2 as cv
import numpy as np
import os
from detectors import SkeletonDetector, BODY_MODEL
from stereo import DisparityCalculator, StereoCapture, StereoParams, prepare_for_vis

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
    if rendered.ndims == 2 or rendered.shape[-1] == 1:
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
    while True:
        ret, (left_frame, right_frame) = cap.read()
        if not ret:
            break

        disparity_map = disp_calc(left_frame, right_frame)

        people = skeleton_detector(left_frame, prepare_for_vis(disparity_map))

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

        display_image = draw_skeleton(left_frame, people)

        cv.imshow(display_image)

        key = cv.waitKey(1)
        if key == 27:
            break
        if key & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
