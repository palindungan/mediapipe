import cv2
import mediapipe as mp
import numpy as np


class MediapipeFaceMesh:
    def __init__(self):
        self.global_color = (0, 255, 0)
        self.img_height, self.img_width, self.img_channel = 0, 0, 0

        # Mediapipe Class
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
            for faceId, faceLandmarks in enumerate(multi_face_landmarks):
                # Draw the face mesh default
                self.mpDrawingSpec.draw_landmarks(img,
                                                  faceLandmarks,
                                                  self.mpFaceMesh.FACEMESH_CONTOURS,
                                                  self.drawingSpec,
                                                  self.drawingSpec)

                # # print landmark
                # for idx, landmark in enumerate(faceLandmarks.landmark):
                #     x, y = int(landmark.x * self.imgWidth), int(landmark.y * self.imgHeight)
                #     print("face: " + str(faceId) + ", id:" + str(idx), ", x:" + str(x), ", y:" + str(y))

        return img

    def drawing_roi(self, img, multi_face_landmarks):
        edges_id = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                    162, 21, 54, 103, 67, 109, 10]
        edges_array = []

        if multi_face_landmarks:
            for faceId, faceLandmarks in enumerate(multi_face_landmarks):
                array = []
                for idx in edges_id:
                    landmark = faceLandmarks.landmark[idx]
                    x, y = int(landmark.x * self.img_width), int(landmark.y * self.img_height)
                    array.append((x, y))
                edges_array.append(array)

        mask = np.zeros_like(img)
        for edges in edges_array:
            points = np.array(edges, np.int32)
            cv2.fillPoly(mask, [points], (255, 255, 255))

        img[~mask.any(axis=2)] = 0

        for edges in edges_array:
            for x, y in edges:
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

        return img
