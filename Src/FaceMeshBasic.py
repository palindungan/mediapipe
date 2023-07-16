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
wCam, hCam = 480, 360  # width and height image
noCam = 1  # default Cam
cameraBrightness = 190  # Set Brightness

globalColor = (255, 0, 0)  # default color
# End of Setting

cap = cv2.VideoCapture(basicTools.getBaseUrl() + "/Resource/Videos/2.mp4")
# cap = cv2.VideoCapture(noCam)
# cap.set(3, wCam)
# cap.set(4, hCam)
# cap.set(10, cameraBrightness)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

while True:
    success, img = cap.read()

    imgOri = img.copy()
    imgBlank = basicTools.CreateBlankImage(img)

    # detect FaceMesh
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
            mpDraw.draw_landmarks(imgBlank, faceLms, mpFaceMesh.FACEMESH_CONTOURS)

    fps = basicTools.countFps(time=time.time())
    cv2.putText(img, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show result in stacked images
    stackedImages = imageProcessing.stackImages(1, ([imgOri, img], [imgBlank, basicTools.CreateBlankImage(img)]))
    cv2.imshow("Stacked Image", stackedImages)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break