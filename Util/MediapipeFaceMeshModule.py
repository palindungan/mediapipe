import cv2
import mediapipe as mp
import numpy as np


class MediapipeFaceMesh:
    def __init__(self):
        self.global_color = (0, 255, 0)
        self.img_height, self.img_width, self.img_channel = 0, 0, 0

        # mediapipe class
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.mpDrawingSpec = mp.solutions.drawing_utils
        self.drawingSpec = self.mpDrawingSpec.DrawingSpec(thickness=1, circle_radius=1, color=self.global_color)

    def processing(self, img):
        self.img_height, self.img_width, self.img_channel = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(img_rgb)
        return results.multi_face_landmarks

    def drawing_img(self, img, multi_face_landmarks):
        if multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(multi_face_landmarks):
                self.mpDrawingSpec.draw_landmarks(img,
                                                  face_landmarks,
                                                  self.mpFaceMesh.FACEMESH_CONTOURS,
                                                  self.drawingSpec,
                                                  self.drawingSpec)

                # # print landmark
                # for idx, landmark in enumerate(face_landmarks.landmark):
                #     x, y = int(landmark.x * self.img_width), int(landmark.y * self.img_height)
                #     print("face: " + str(face_idx) + ", id:" + str(idx), ", x:" + str(x), ", y:" + str(y))

        return img

    def get_outer_edge(self, multi_face_landmarks):
        outer_edges = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                       361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                       176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                       162, 21, 54, 103, 67, 109, 10]
        multi_face_coordinates = []

        if multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(multi_face_landmarks):
                coordinate = []
                for edge_idx in outer_edges:
                    landmark = face_landmarks.landmark[edge_idx]
                    x, y = int(landmark.x * self.img_width), int(landmark.y * self.img_height)
                    coordinate.append((x, y))
                multi_face_coordinates.append(coordinate)

                print("face: " + str(face_idx) + ", x:" + str(x), ", y:" + str(y))

        return multi_face_coordinates

    def drawing_roi(self, img, multi_face_landmarks):
        outer_edges = self.get_outer_edge(multi_face_landmarks)

        mask = np.zeros_like(img)  # create blank image

        # draw face mask (roi / true = white)
        for face_idx, edges in enumerate(outer_edges):
            points = np.array(edges, np.int32)
            cv2.fillPoly(mask, [points], (255, 255, 255))

        img[~mask.any(axis=2)] = 0  # replace rgb 0 based on ~mask

        # draw bboxes
        multi_face_bboxes = []
        for face_idx, edges in enumerate(outer_edges):
            x_list = []
            y_list = []

            for x, y in edges:
                x_list.append(x)
                y_list.append(y)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            padding = 20
            bbox = x_min - padding, y_min - padding, x_max + padding, y_max + padding
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.global_color, 2)

            multi_face_bboxes.append(bbox)

        return img
