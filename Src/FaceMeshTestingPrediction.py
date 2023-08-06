import cv2
import time
import pickle
import numpy as np
import os

from keras_facenet import FaceNet

from Util import BasicToolModule
from Util import ImageProcessingModule
from Util import MediapipeFaceMeshModule

# utility class
basicTool = BasicToolModule.BasicTool()
imageProcessing = ImageProcessingModule.ImageProcessing()
mediapipeFaceMesh = MediapipeFaceMeshModule.MediapipeFaceMesh()

# camera setting
w_cam, h_cam = 480, 360  # width and height image
no_cam = 1  # default Cam
camera_brightness = 190  # set brightness
global_color = (0, 255, 0)  # default color

cap = cv2.VideoCapture(basicTool.get_base_url() + "/Resource/Videos/1.mp4")  # read file

# # webcam
# cap = cv2.VideoCapture(no_cam)  # webcam
# cap.set(3, w_cam)  # width
# cap.set(4, h_cam)  # height
# cap.set(10, camera_brightness)  # brightness

face_net = FaceNet()

model_file = open(basicTool.get_base_url() + '/Resource/' + 'data.pkl', "rb")
model_database = pickle.load(model_file)
model_file.close()

while True:
    success, img = cap.read()
    img_ori = img.copy()

    # processing
    multi_face_landmarks = mediapipeFaceMesh.processing(img)
    get_roi_images = mediapipeFaceMesh.get_roi_images(img, multi_face_landmarks)

    for roi_idx, roi_image in enumerate(get_roi_images):
        face = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))

        face = np.expand_dims(face, axis=0)
        signature = face_net.embeddings(face)

        min_dist = 100
        identity = ' '
        for key, value in model_database.items():
            dist = np.linalg.norm(value - signature)
            if dist < min_dist:
                min_dist = dist
                identity = key

        cv2.putText(img_ori, identity, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # show fps
    fps = basicTool.count_fps(my_time=time.time())
    cv2.putText(img_ori, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, global_color, 3)

    # show images in stacked
    stacked_images = imageProcessing.stack_images(1, ([img_ori]))
    cv2.imshow("Stacked Image", stacked_images)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
