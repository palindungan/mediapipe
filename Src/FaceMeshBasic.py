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
    result = faceMesh.process(imgRGB)

    fps = basicTools.countFps(time=time.time())
    cv2.putText(img, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    cv2.imshow("Image", img)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break