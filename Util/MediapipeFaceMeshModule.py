import cv2
import mediapipe as mp
import numpy as np

from Util import BasicToolModule


class MediapipeFaceMesh:
    def __init__(self):
        self.globalColor = (0, 255, 0)  # default color

        self.basicTools = BasicToolModule.BasicTool()

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.mpDrawingSpec = mp.solutions.drawing_utils
        self.drawingSpec = self.mpDrawingSpec.DrawingSpec(thickness=1, circle_radius=1, color=self.globalColor)

    def processing(self, img):
        imgContour = self.basicTools.CreateBlankImage(img)
        imgROI = img.copy()

        face_edges_list = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        multiFaceLandmarks = results.multi_face_landmarks
        if multiFaceLandmarks:
            for faceId, faceLandmarks in enumerate(multiFaceLandmarks):
                # Draw the face mesh default
                self.mpDrawingSpec.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS,
                                                  self.drawingSpec, self.drawingSpec)
                self.mpDrawingSpec.draw_landmarks(imgContour, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS,
                                                  self.drawingSpec, self.drawingSpec)

                # print landmark
                for idx, lm in enumerate(faceLandmarks.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    print("face: " + str(faceId) + ", id:" + str(idx), ", x:" + str(x), ", y:" + str(y))

                face_edges = []
                outer_edges = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                               361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                               176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                               162, 21, 54, 103, 67, 109, 10]
                for idx in outer_edges:
                    landmark = faceLandmarks.landmark[idx]
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    face_edges.append((x, y))
                face_edges_list.append(face_edges)

        mask = np.zeros_like(imgROI)
        for face_edges in face_edges_list:
            points = np.array(face_edges, np.int32)
            cv2.fillPoly(mask, [points], (255, 255, 255))

        imgROI[~mask.any(axis=2)] = 0

        for face_edges in face_edges_list:
            for x, y in face_edges:
                cv2.circle(imgROI, (x, y), 1, (0, 255, 0), -1)

        return img, imgContour, imgROI
