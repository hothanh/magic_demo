import cv2 as cv
import numpy as np
from ._output_convert import get_humans


BODY_MODEL = [
    (0, [1]),           # head
    (1, [0, 2, 3, 8]),  # neck
    (2, [1, 4]),        # r shoulder
    (3, [1, 5]),        # l shoulder
    (4, [2, 6]),        # r elbow
    (5, [3, 7]),        # l elbow
    (6, [4]),           # r hand
    (7, [5]),           # l hand
    (8, [1, 9, 10]),    # torso
    (9, [8, 11]),       # r hip
    (10, [8, 12]),      # l hip
    (11, [9, 13]),      # r knee
    (12, [10, 14]),     # l knee
    (13, [11]),         # r foot
    (14, [12])          # l foot
]

BODY_EDGES = []
for i, js in BODY_MODEL:
    for j in js:
        edge = (min(i, j), max(i, j))
        if edge not in BODY_EDGES:
            BODY_EDGES.append(edge)


class SkeletonDetector:
    def __init__(self, pb_model_path, image_size, n_threads=None):
        self._net = cv.dnn.readNetFromTensorflow(pb_model_path)
        self._net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self._image_size = (image_size[0], image_size[1])
        self._n_threads = n_threads

    def __call__(self, gray_image, ret_maps=False):
        orig_shape = gray_image.shape

        gray_image = cv.resize(gray_image, self._image_size, interpolation=cv.INTER_LINEAR)

        blob = cv.dnn.blobFromImage(gray_image, 1, self._image_size)
        self._net.setInput(blob, 'model/inputs/gray')

        old = cv.getNumThreads()
        if self._n_threads is not None:
            cv.setNumThreads(self._n_threads)

        joints_map, bones_map = self._net.forward(['model/joints_map/Sigmoid', 'model/bones_map/Conv2D'])

        cv.setNumThreads(old)

        joints_map = np.transpose(joints_map, [0, 2, 3, 1])[0]

        bones_map = np.transpose(bones_map, [0, 2, 3, 1])[0]
        bones_map = np.reshape(bones_map, [bones_map.shape[0], bones_map.shape[1], bones_map.shape[2]//2, 2])

        people = get_humans(joints_map, bones_map, BODY_EDGES, BODY_MODEL)
        people = [
            [((joint[0] * orig_shape[1] / joints_map.shape[1],  joint[1] * orig_shape[0] / joints_map.shape[0]) if joint is not None else None)
             for joint in person] for person in people
        ]
        if not ret_maps:
            return people
        else:
            joints_map = cv.resize(joints_map, (orig_shape[1], orig_shape[0]), interpolation=cv.INTER_LINEAR)
            bones_map = np.reshape(bones_map, [bones_map.shape[0], bones_map.shape[1], bones_map.shape[2]*2])
            bones_map = cv.resize(bones_map, (orig_shape[1], orig_shape[0]), interpolation=cv.INTER_LINEAR)
            bones_map = np.reshape(bones_map, [bones_map.shape[0], bones_map.shape[1], bones_map.shape[2]//2, 2])
            return people, (joints_map, bones_map)
