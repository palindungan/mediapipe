import cv2
import mediapipe as mp
import time
from Util import BasicToolModule
from Util import ImageProcessingModule

# Start of Declare Object Class
basicTools = BasicToolModule.BasicTools()
imageProcessing = ImageProcessingModule.ImageProcessing()
# End of Declare Object Class

# Start of Setting
globalColor = (255, 0, 0)  # default color
# End of Setting

cap = cv2.VideoCapture(basicTools.getBaseUrl() + "/Resource/Videos/2.mp4")

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

while True:
    success, img = cap.read()

    # detect FaceMesh
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # if results.multi_face_landmarks:
    #     for faceLms in results.multi_face_landmarks:
    #         mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Draw a circle for each face landmark point
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    fps = basicTools.countFps(time=time.time())
    cv2.putText(img, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    cv2.imshow("Image", img)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break