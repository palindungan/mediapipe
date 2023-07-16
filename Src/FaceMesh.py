import cv2
import mediapipe as mp
import time
import pandas as pd
from Util import BasicToolModule
from Util import ImageProcessingModule


def getFaceOval():
    face_oval = mpFaceMesh.FACEMESH_FACE_OVAL
    df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])
    print(df)

    routes_idx = []

    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):
        # print(p1, p2)

        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]

        route_idx = []
        route_idx.append(p1)
        route_idx.append(p2)
        routes_idx.append(route_idx)

    # -------------------------------

    for route_idx in routes_idx:
        print(f"Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point")


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

getFaceOval()

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
                # print("face: " + str(faceId) + ", id:" + str(idx), ", x:" + str(x), ", y:" + str(y))

    fps = basicTools.countFps(time=time.time())
    cv2.putText(imgOri, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, globalColor, 3)

    # show result in stacked images
    stackedImages = imageProcessing.stackImages(1, ([imgOri, imgContours], [img, basicTools.CreateBlankImage(img)]))
    cv2.imshow("Stacked Image", stackedImages)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
