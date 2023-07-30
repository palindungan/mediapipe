import mediapipe as mp


class MediapipeFaceMesh:
    def __init__(self):
        self.globalColor = (0, 255, 0)  # default color

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.mpDrawingSpec = mp.solutions.drawing_utils
        self.drawingSpec = self.mpDrawingSpec.DrawingSpec(thickness=1, circle_radius=1, color=self.globalColor)
