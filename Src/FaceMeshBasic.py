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
##################
wCam, hCam = 480, 360  # width and height image
noCam = 1  # default Cam
globalColor = (255, 0, 0)  # default color
detectionCon = 0.70  # set Confident in AI Mediapipe

cameraBrightness = 190  # Set Brightness
moduleVal = 5  # SAVE EVERY 1 FRAME TO AVOID REPETITION
minBlur = 500  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
grayImage = False  # IMAGE SAVED COLORED OR GRAY
saveData = False  # SAVE DATA FLAG
imgWidth = 180  # Resize width Image
imgHeight = 120  # Resize height Image
##################
# End of Setting

cap = cv2.VideoCapture(basicTools.getBaseUrl() + "/Resource/Videos/2.mp4")

while True:
    success, img = cap.read()

    fps = basicTools.countFps(time=time.time())
    cv2.putText(img, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)