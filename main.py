import config
import cv2 as cv
import numpy as np
import os
from detectors import SkeletonDetector, BODY_MODEL
from stereo import DisparityCalculator, StereoCapture, StereoParams, prepare_for_vis
import time
import random

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


def min_max_norm(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn)


def get_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xy1 = np.array([x1, y1], np.float32)
    wh1 = np.array([w1, h1], np.float32)

    xy2 = np.array([x2, y2], np.float32)
    wh2 = np.array([w2, h2], np.float32)

    mins1 = xy1
    maxes1 = xy1 + wh1

    mins2 = xy2
    maxes2 = xy2 + wh2

    intersect_mins = np.maximum(mins1, mins2)
    intersect_maxes = np.minimum(maxes1, maxes2)
    intersect_wh = np.maximum(0, intersect_maxes - intersect_mins)

    intersect_areas = np.prod(intersect_wh)

    areas1 = np.prod(wh1)
    areas2 = np.prod(wh2)

    union_areas = areas1 + areas2 - intersect_areas

    iou = intersect_areas / np.maximum(union_areas, 1e-6)

    return iou


def main():
    intrinsics_path = os.path.join(ROOT_DIR, 'models', 'intrinsics.yml')
    if not os.path.exists(intrinsics_path):
        raise RuntimeError('Can\'t find a intrinsics file!')

    extrinsics_path = os.path.join(ROOT_DIR, 'models', 'extrinsics.yml')
    if not os.path.exists(extrinsics_path):
        raise RuntimeError('Can\'t find a extrinsics file!')

    skeleton_model_path = os.path.join(ROOT_DIR, 'models', 'pose-unet-96x128.pb')
    if not os.path.exists(skeleton_model_path):
        raise RuntimeError('Can\'t find a skeleton detector model!')

    # facial_landmarks_model_path = os.path.join(ROOT_DIR, 'models', 'shape_predictor_68_face_landmarks.dat')
    # if not os.path.exists(facial_landmarks_model_path):
    #     raise RuntimeError('Can\'t find a facial landmark detection model!')

    # age_pca_path = os.path.join(ROOT_DIR, 'models', 'agedb-pca.pkl')
    # if not os.path.exists(age_pca_path):
    #     raise RuntimeError('Can\'t find an age pca model!')

    # gender_svm_path = os.path.join(ROOT_DIR, 'models', 'agedb-gender_svm.pkl')
    # if not os.path.exists(gender_svm_path):
    #     raise RuntimeError('Can\'t find a gender svm model!')

    # age_svm_path = os.path.join(ROOT_DIR, 'models', 'agedb-svm.pkl')
    # if not os.path.exists(age_svm_path):
    #     raise RuntimeError('Can\'t find an age svm model!')

    # age_svr_path = os.path.join(ROOT_DIR, 'models', 'agedb-svr.pkl')
    # if not os.path.exists(age_svr_path):
    #     raise RuntimeError('Can\'t find an age svr model!')

    # age_features_stat_path = os.path.join(ROOT_DIR, 'models', 'agedb-mean_std.npz')
    # if not os.path.exists(age_features_stat_path):
    #     raise RuntimeError('Can\'t find an age features statistics file!')

    skeleton_detector = SkeletonDetector(skeleton_model_path, (128, 96), 8)

    # age_gender_predictor = AgeGenderPredictor(
    #     facial_landmarks_model_path, age_pca_path, gender_svm_path, age_svm_path, age_svr_path, age_features_stat_path)

    stereo_params = StereoParams(intrinsics_path, extrinsics_path)

    if config.ENABLE_DISPARITY:
        disp_calc = DisparityCalculator(**config.SGBM_params)

    if isinstance(config.VIDEO_SOURCE, int):
        cap = StereoCapture(config.VIDEO_SOURCE, stereo_params)
    else:
        cap = StereoCapture(config.VIDEO_SOURCE)

    print('Capture: (%i, %i) %.2f' % (int(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), cap.get(cv.CAP_PROP_FPS)), flush=True)

    if config.DEBUG_WRITE:
        writer = 0
        if config.ENABLE_DISPARITY:
            disp_writer = None
        left_writer = None
        right_writer = None

    try:
        print('Processing is started.', flush=True)

        people = []
        last_id = 0
        key = None

        start = time.time()
        while True:
            cap_start = time.time()
            ret, (left_frame, right_frame) = cap.read()
            cap_elapsed = time.time() - cap_start
            if not ret:
                break

            # disparity calculation

            if config.ENABLE_DISPARITY:
                disp_start = time.time()
                disparity_map = disp_calc(left_frame, right_frame)
                disp_elapsed = time.time() - disp_start
            else:
                disp_elapsed = 1e-3

            # skeleton detection

            skeleton_start = time.time()
            frame_people_skeletons, (joints_map, bones_map) = skeleton_detector(left_frame, ret_maps=True)
            skeleton_elapsed = time.time() - skeleton_start

            # tracking

            tracking_start = time.time()

            frame_people = []
            for person in frame_people_skeletons:
                head = person[0]
                neck = person[1]

                if head is not None and neck is not None:
                    dist = np.linalg.norm(np.array(head)-neck) * 1.25
                    bbox = [head[0]-dist, head[1]-dist, head[0]+dist, head[1]+dist]

                    frame_people.append((bbox, person))

            if not people:
                people = list(zip(range(last_id, len(frame_people)), frame_people, [None]*len(frame_people)))
                last_id = last_id + len(frame_people)
            else:
                new_people = []
                if frame_people:
                    for i in range(len(people)):
                        person = people[i]

                        best_j = None
                        best_iou = 0
                        for j in range(len(frame_people)):
                            iou = get_iou(person[1][0], frame_people[j][0])
                            if max(0, iou - 0.5) > 0 and best_iou < iou:
                                best_iou = iou
                                best_j = j

                        if best_j is not None:
                            new_people.append((person[0], frame_people[best_j], person[2]))
                            del frame_people[best_j]

                    if new_people:
                        last_id = max(new_people, key=lambda x: x[0])[0]+1
                    for person in frame_people:
                        new_people.append((last_id, person, None))
                        last_id += 1

                people = new_people

            tracking_elapsed = time.time() - tracking_start

            # age & gender detection

            if people and key is not None and key & 0xFF == ord('d'):
                # age_gender_start = time.time()

                # _, people_data, _ = zip(*people)
                # people_bboxes, _ = zip(*people_data)

                # predictions = age_gender_predictor(left_frame, people_bboxes)

                # for i, gender, age in predictions:
                #     people[i] = (people[i][0], people[i][1], (gender, age))

                age_gender_elapsed = 1e-3  # time.time() - age_gender_start
            else:
                age_gender_elapsed = 1e-3

            end = time.time()
            elapsed = end - start
            frame_elapsed = end - cap_start
            start = end

            print('\rFPS: %.2f; CAP: %.2f; DISP: %.2f; SKLTN: %.2f; TRACK: %.2f; AGE: %.2f; FRAME: %.2f, PPL_FOUND: %i' %
                  (1/elapsed, 1/cap_elapsed, 1/disp_elapsed, 1/skeleton_elapsed, 1/tracking_elapsed, 1/age_gender_elapsed, 1/frame_elapsed, len(people)), end='', flush=True)

            display_image = cv.cvtColor(left_frame, cv.COLOR_GRAY2BGR)

            display_image[..., 2] = np.uint8(
                255*np.clip((np.float32(display_image[..., 2]) / 255) + 0.9*min_max_norm(np.max(joints_map, axis=-1)), 0, 1))
            display_image[..., 1] = np.uint8(
                255*np.clip((np.float32(display_image[..., 1]) / 255) + 0.5*min_max_norm(np.max(np.linalg.norm(bones_map, axis=-1), axis=-1)), 0, 1))

            if people:
                _, people_data, _ = zip(*people)
                _, skeletons = zip(*people_data)

                display_image = draw_skeleton(display_image, skeletons)
                for id_, (bbox, _), extra in people:
                    cv.rectangle(display_image, (int(bbox[0]), int(bbox[1])),
                                 (int(bbox[2]), int(bbox[3])), (32, 32, 225))

                    person_line = ('#%i' % id_)
                    if extra is not None:
                        gender, age = extra

                        person_line = '%s a: %i' % (person_line, age)
                        person_line = '%s %s' % (person_line, 'M' if gender == 1 else 'F')

                    cv.putText(display_image, person_line, (int(bbox[0])+5, int(bbox[1])+16),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (160, 32, 225), 2)

            cv.imshow('demo', display_image)
            if config.ENABLE_DISPARITY:
                cv.imshow('disparity', prepare_for_vis(disparity_map))

            if config.DEBUG_WRITE:
                if isinstance(writer, int):
                    writer += 1
                    if writer > 2:
                        writer = cv.VideoWriter('./demo.mp4', cv.VideoWriter_fourcc(*'avc1'), 1/elapsed, (int(
                            cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
                        if config.ENABLE_DISPARITY:
                            disp_writer = cv.VideoWriter('./demo-disps.mp4', cv.VideoWriter_fourcc(*'avc1'), 1/elapsed, (int(
                                cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), False)
                        left_writer = cv.VideoWriter('./demo-lefts.mp4', cv.VideoWriter_fourcc(*'avc1'), 1/elapsed, (int(
                            cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), False)
                        right_writer = cv.VideoWriter('./demo-rights.mp4', cv.VideoWriter_fourcc(*'avc1'), 1/elapsed, (int(
                            cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))), False)
                else:
                    writer.write(display_image)
                    if config.ENABLE_DISPARITY:
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
                if config.ENABLE_DISPARITY:
                    disp_writer.release()
                left_writer.release()
                right_writer.release()


if __name__ == "__main__":
    main()
