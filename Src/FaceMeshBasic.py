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

globalColor = (0, 255, 0)  # default color
# End of Setting

cap = cv2.VideoCapture(basicTools.getBaseUrl() + "/Resource/Videos/3.mp4")
# cap = cv2.VideoCapture(noCam)
# cap.set(3, wCam)
# cap.set(4, hCam)
# cap.set(10, cameraBrightness)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=globalColor)

while True:
    success, img = cap.read()

    imgOri = img.copy()
    imgContours = basicTools.CreateBlankImage(img)

    # detect FaceMesh
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceId, faceLms in enumerate(results.multi_face_landmarks):
            # Draw the face mesh default
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            mpDraw.draw_landmarks(imgContours, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            # Draw the face mesh manual
            # for idx, landmark in enumerate(faceLms.landmark):
            #     x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            #     cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
            #     cv2.circle(imgContours, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

            # print landmark
            for idx, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print("face: " + str(faceId) + ", id:" + str(idx), ", x:" + str(x), ", y:" + str(y))

    fps = basicTools.countFps(time=time.time())
    cv2.putText(imgOri, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show result in stacked images
    stackedImages = imageProcessing.stackImages(1, ([imgOri, imgContours], [img, basicTools.CreateBlankImage(img)]))
    cv2.imshow("Stacked Image", stackedImages)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


def draw_face_shape(image, face_landmarks):
    # Gambar poligon untuk menggambarkan bentuk wajah
    # Bentuk wajah yang diambil berdasarkan landmark Mediapipe
    face_contour = [
        face_landmarks[0], face_landmarks[1], face_landmarks[2], face_landmarks[3],
        face_landmarks[4], face_landmarks[5], face_landmarks[6], face_landmarks[7],
        face_landmarks[8], face_landmarks[9], face_landmarks[10], face_landmarks[11],
        face_landmarks[12], face_landmarks[13], face_landmarks[14], face_landmarks[15],
        face_landmarks[16], face_landmarks[17], face_landmarks[18], face_landmarks[19],
        face_landmarks[20], face_landmarks[21], face_landmarks[22], face_landmarks[23],
        face_landmarks[24], face_landmarks[25], face_landmarks[26], face_landmarks[27],
        face_landmarks[28], face_landmarks[29], face_landmarks[30], face_landmarks[31],
        face_landmarks[32], face_landmarks[33], face_landmarks[34], face_landmarks[35],
        face_landmarks[36], face_landmarks[39], face_landmarks[40], face_landmarks[41],
        face_landmarks[42], face_landmarks[43], face_landmarks[44], face_landmarks[45],
        face_landmarks[46], face_landmarks[47], face_landmarks[48], face_landmarks[49],
        face_landmarks[50], face_landmarks[51], face_landmarks[52], face_landmarks[53],
        face_landmarks[54], face_landmarks[55], face_landmarks[56], face_landmarks[57],
        face_landmarks[58], face_landmarks[59], face_landmarks[60], face_landmarks[61],
        face_landmarks[62], face_landmarks[63], face_landmarks[64], face_landmarks[65],
        face_landmarks[66], face_landmarks[67]
    ]

    # Gambar bentuk wajah dengan poligon
    cv2.polylines(image, [face_contour], isClosed=True, color=(0, 255, 0), thickness=2)

    return image
