import cv2
import time

from Util import BasicToolModule
from Util import ImageProcessingModule
from Util import MediapipeFaceMeshModule

# Utility Class
basicTools = BasicToolModule.BasicTool()
imageProcessing = ImageProcessingModule.ImageProcessing()
mediapipeFaceMesh = MediapipeFaceMeshModule.MediapipeFaceMesh()

# Camera Setting
wCam, hCam = 480, 360  # width and height image
noCam = 1  # default Cam
cameraBrightness = 190  # set brightness
globalColor = (0, 255, 0)  # default color

# cap = cv2.VideoCapture(basicTools.getBaseUrl() + "/Resource/Videos/3.mp4")  # read file

# Webcam Video
cap = cv2.VideoCapture(noCam)  # webcam
cap.set(3, wCam)  # width
cap.set(4, hCam)  # height
cap.set(10, cameraBrightness)  # brightness

while True:
    success, img = cap.read()
    imgOri = img.copy()

    # processing
    multiFaceLandmarks = mediapipeFaceMesh.processing(img)
    img, imgContour, imgROI = mediapipeFaceMesh.drawing(img, multiFaceLandmarks)

    # show fps
    fps = basicTools.countFps(time=time.time())
    cv2.putText(imgOri, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show images in stacked
    stackedImages = imageProcessing.stackImages(1, ([imgOri, imgContour], [img, imgROI]))
    cv2.imshow("Stacked Image", stackedImages)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
