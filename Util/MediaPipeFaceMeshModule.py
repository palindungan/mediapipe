import cv2
import mediapipe as mp
import time
from Util import BasicToolModule
from Util import ImageProcessingModule


class MediaPipeFaceMesh():

    def __init__(self, staticMode=False, maxFaces=1, refineLandmarks=False, minDetectionCon=0.5, minTrackingCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        self.globalColor = (0, 255, 0)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refineLandmarks,
                                                 self.minDetectionCon, self.minTrackingCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=self.globalColor)

    def detection(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []

        if results.multi_face_landmarks:
            for faceId, faceLms in enumerate(results.multi_face_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,
                                               self.drawSpec)
                # print landmark
                face = []
                for idx, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # print("face: " + str(faceId) + ", id:" + str(idx), ", x:" + str(x), ", y:" + str(y))
                    # cv2.putText(img, f'FPS {int(idx)}', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, self.globalColor, 1)
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():
    # Start of Declare Object Class
    basicTools = BasicToolModule.BasicTools()
    imageProcessing = ImageProcessingModule.ImageProcessing()
    mpFaceMesh = MediaPipeFaceMesh(maxFaces=2)
    # End of Declare Object Class

    wCam, hCam = 1280, 720  # width and height image
    noCam = 1  # default Cam
    cameraBrightness = 190  # Set Brightness
    globalColor = (0, 255, 0)  # default color
    # cap = cv2.VideoCapture(basicTools.getBaseUrl() + "/Resource/Videos/3.mp4")
    cap = cv2.VideoCapture(noCam)
    cap.set(3, wCam)
    cap.set(4, hCam)
    cap.set(10, cameraBrightness)

    while True:
        success, img = cap.read()

        imgOri = img.copy()
        imgContours = basicTools.CreateBlankImage(img)

        fps = basicTools.countFps(time=time.time())
        cv2.putText(imgOri, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

        # show result in stacked images
        stackedImages = imageProcessing.stackImages(1, ([imgOri, imgContours], [img, basicTools.CreateBlankImage(img)]))
        cv2.imshow("Stacked Image", stackedImages)

        # action for end proses
        if cv2.waitKey(1) & 0xff == ord('q'):
            break


if __name__ == '__main__':
    main()
