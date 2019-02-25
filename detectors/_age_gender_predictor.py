import cv2 as cv
import dlib
import pickle
import numpy as np
from skimage.feature import local_binary_pattern

TARGET_DIST = 50
P = 8
R = 3


def get_region_coordinates(landmarks):
    regions = [
        (landmarks[48, 0] - 15, landmarks[48, 1] + 7,  landmarks[48, 0] + 5,  landmarks[48, 1] - 7),
        (landmarks[54, 0] - 5,  landmarks[54, 1] + 7,  landmarks[54, 0] + 15, landmarks[54, 1] - 7),
        (landmarks[36, 0] - 20, landmarks[36, 1] + 10, landmarks[36, 0],      landmarks[36, 1] - 10),
        (landmarks[45, 0],      landmarks[45, 1] + 10, landmarks[45, 0] + 20, landmarks[45, 1] - 10),
        (landmarks[31, 0] - 25, landmarks[31, 1] + 12, landmarks[31, 0] - 5,  landmarks[31, 1] - 12),
        (landmarks[35, 0] + 5,  landmarks[35, 1] + 12, landmarks[35, 0] + 25, landmarks[35, 1] - 12),
        (landmarks[41, 0] - 12, landmarks[41, 1] + 30, landmarks[41, 0] + 18, landmarks[41, 1]),
        (landmarks[46, 0] - 18, landmarks[46, 1] + 30, landmarks[46, 0] + 12, landmarks[46, 1]),
        (landmarks[27, 0] - 12, landmarks[27, 1] + 25, landmarks[27, 0] + 12, landmarks[27, 1] - 20),
        (landmarks[57, 0] - 22, landmarks[57, 1] + 25, landmarks[57, 0] + 22, landmarks[57, 1] + 3),
        (landmarks[27, 0] - 35, landmarks[27, 1] - 35, landmarks[27, 0] + 35, landmarks[27, 1] - 65),
    ]

    return regions


def get_skin_areas(img, face_rects, landmark_predictor):
    image_skin_areas = []
    for rectangle in face_rects:
        # Детектируем ландмарки
        landmarks = landmark_predictor(img, rectangle)
        landmarks = np.array([(l.x, l.y) for l in landmarks.parts()], np.int32)

        bbox = np.array([rectangle.left(), rectangle.top(), rectangle.right(), rectangle.bottom()], np.int32)

        left_point = landmarks[39]
        right_point = landmarks[42]

        center = (left_point + right_point) * 0.5

        # Выравниваем поворот изображения
        dist = np.linalg.norm(left_point - right_point)
        sin_alpha = (right_point[1] - left_point[1]) / dist
        alpha_rad = np.arcsin(sin_alpha)
        alpha_deg = alpha_rad * 180 / np.pi

        scale = TARGET_DIST / dist

        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]]
        ], np.float32)

        R = np.array([
            [np.cos(alpha_rad), np.sin(alpha_rad)],
            [-np.sin(alpha_rad), np.cos(alpha_rad)]
        ])

        points = np.int32((points - center) @ R + center)

        bbox_larger = np.maximum(0, np.min(points, axis=0)), np.maximum(0, np.max(points, axis=0))

        w, h = bbox_larger[1] - bbox_larger[0]

        face = img[bbox_larger[0][1]:bbox_larger[1][1], bbox_larger[0][0]:bbox_larger[1][0]]
        center = center - bbox_larger[0]
        landmarks = landmarks - bbox_larger[0]

        M = cv.getRotationMatrix2D((int(center[0]), int(center[1])), alpha_deg, 1)

        transformed_img = cv.warpAffine(face, M, (w, h))
        transformed_img = cv.resize(transformed_img, (int(w*scale), int(h*scale)), interpolation=cv.INTER_LINEAR)

        # Трансформируем ключевые точки
        transform_martix = M[:, :2]
        shift_matrix = M[:, 2]

        new_landmarks = landmarks @ transform_martix.T + shift_matrix
        new_landmarks = new_landmarks.astype(np.int32)

        # Выжедяем регионы
        regions = get_region_coordinates(new_landmarks)
        skin_areas = []
        for region in regions:
            area = transformed_img[region[3]:region[1], region[0]:region[2]]
            skin_areas.append(area)

        image_skin_areas.append((rectangle, skin_areas))

    return image_skin_areas


def get_lbp_features(image_skin_areas, p=8, r=1):
    results = []
    for i, (_, skin_areas) in enumerate(image_skin_areas):
        try:
            patterns = []

            lbp_hist = np.zeros(shape=(256,))
            for area in skin_areas:
                if area.size:
                    lbp = local_binary_pattern(area, p, r, 'nri_uniform')
                    lbp_hist = np.bincount(lbp.flatten().astype(np.int32), minlength=59)
                else:
                    lbp_hist = np.zeros([59])
                patterns.append(lbp_hist)

            features = np.concatenate(patterns)
            results.append((i, features))
        except Exception as ex:
            print(repr(ex))
            continue

    return results


class AgeGenderPredictor:
    def __init__(self, facial_landmarks_model_path, age_pca_path, gender_svm_path, age_svm_path, age_svr_path, age_features_stat_path):
        self._landmark_predictor = dlib.shape_predictor(facial_landmarks_model_path)

        with np.load(age_features_stat_path) as mean_std:
            self._mean = mean_std['mean']
            self._std = mean_std['std']

        with open(age_pca_path, 'rb') as f:
            self._pca = pickle.load(f)

        with open(age_svm_path, 'rb') as f:
            self._age_clasifier = pickle.load(f)

        with open(age_svr_path, 'rb') as f:
            self._age_svr = pickle.load(f)

        with open(gender_svm_path, 'rb') as f:
            self._gender_svm = pickle.load(f)

    def __call__(self, frame, people_bboxes):
        face_rects = [
            dlib.rectangle(max(0, int(bbox[0])), max(0, int(bbox[1])), max(0, int(bbox[2])), max(0, int(bbox[3])))
            for bbox in people_bboxes
        ]

        image_skin_areas = get_skin_areas(frame, face_rects, self._landmark_predictor)
        lbp_features = get_lbp_features(image_skin_areas, P, R)
        if not lbp_features:
            return []

        face_indices, features = zip(*lbp_features)

        try:
            features = self._pca.transform(features)
            features -= self._mean
            features /= self._std

            classes = self._age_clasifier.predict(features)

            ages = []
            for i, _class in enumerate(classes):
                age = self._age_svr[_class].predict([features[i]])
                ages.append(age[0])

            genders = self._gender_svm.predict(features)

            return zip(face_indices, genders, ages)
        except:
            return []
