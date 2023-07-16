import cv2
import time
from Util import BasicToolModule
from Util import ImageProcessingModule

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

# Start of Declare Object Class
basicTools = BasicToolModule.BasicTools()
imageProcessing = ImageProcessingModule.ImageProcessing()
# End of Declare Object Class

# Start of Set
cap = cv2.VideoCapture(noCam)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(10, cameraBrightness)

myPath = basicTools.getBaseUrl() + '/Resource/dataset/numeric/'  # PATH TO SAVE IMAGE

countSave = 0
# End of Set

# create folder for new dataset
if saveData:
    basicTools.CreateDirectory(myPath)

while True:
    # read image from cam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # flip the image
    imgCopy = img.copy()

    # detect face mash

    imgRoi = cv2.resize(img, (wCam, hCam))  # resize img region of interest

    # show text
    fps = basicTools.countFps(time=time.time())
    cv2.putText(img, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)
    cv2.putText(img, f'Saved : {int(countSave)}', (170, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show result in stacked images
    stackedImages = imageProcessing.stackImages(1, ([img, basicTools.CreateBlankImage(img)], [basicTools.CreateBlankImage(img), basicTools.CreateBlankImage(img)]))
    cv2.imshow("Stacked Image", stackedImages)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
