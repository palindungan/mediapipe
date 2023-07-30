import cv2
import mediapipe as mp
import time
import numpy as np

from Util import BasicToolModule
from Util import ImageProcessingModule

# Utility Class
basicTools = BasicToolModule.BasicTools()
imageProcessing = ImageProcessingModule.ImageProcessing()

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

# mediapipe FaceMesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=globalColor)

while True:
    success, img = cap.read()

    imgOri = img.copy()
    imgContours = basicTools.CreateBlankImage(img)
    imgROI = img.copy()

    # detect FaceMesh
    face_edges_list = []

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceId, faceLms in enumerate(results.multi_face_landmarks):
            # Draw the face mesh default
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            mpDraw.draw_landmarks(imgContours, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

            # print landmark
            for idx, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print("face: " + str(faceId) + ", id:" + str(idx), ", x:" + str(x), ", y:" + str(y))

            face_edges = []
            outer_edges = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                           361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                           176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                           162, 21, 54, 103, 67, 109, 10]
            for idx in outer_edges:
                landmark = faceLms.landmark[idx]
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                face_edges.append((x, y))
            face_edges_list.append(face_edges)

    # Gambar poligon wajah terluar pada semua wajah
    mask = np.zeros_like(imgROI)
    for face_edges in face_edges_list:
        points = np.array(face_edges, np.int32)
        cv2.fillPoly(mask, [points], (255, 255, 255))

    # Gambar hitam di luar poligon wajah terluar
    imgROI[~mask.any(axis=2)] = 0

    # Gambar titik-titik tepi wajah terluar pada semua wajah
    for face_edges in face_edges_list:
        for x, y in face_edges:
            cv2.circle(imgROI, (x, y), 1, (0, 255, 0), -1)

    fps = basicTools.countFps(time=time.time())
    cv2.putText(imgOri, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show result in stacked images
    stackedImages = imageProcessing.stackImages(1, ([imgOri, imgContours], [img, imgROI]))
    cv2.imshow("Stacked Image", stackedImages)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
