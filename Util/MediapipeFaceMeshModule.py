import mediapipe as mp


class MediapipeFaceMesh:
    def __init__(self):
        globalColor = (0, 255, 0)  # default color

        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
        mpDrawingSpec = mp.solutions.drawing_utils
        drawingSpec = mpDrawingSpec.DrawingSpec(thickness=1, circle_radius=1, color=globalColor)
