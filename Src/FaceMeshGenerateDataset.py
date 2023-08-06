import cv2
import time
import numpy as np

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

cap = cv2.VideoCapture(basicTool.get_base_url() + "/Resource/Videos/3.mp4")  # read file

# # webcam
# cap = cv2.VideoCapture(no_cam)  # webcam
# cap.set(3, w_cam)  # width
# cap.set(4, h_cam)  # height
# cap.set(10, camera_brightness)  # brightness

path_save = basicTool.get_base_url() + '/Resource/Dataset/'  # PATH TO SAVE IMAGE
count_saved = 0
is_save_data = True

if is_save_data:
    basicTool.create_directory(path_save)

while True:
    success, img = cap.read()
    img_ori = img.copy()

    # processing
    multi_face_landmarks = mediapipeFaceMesh.processing(img)

    # get roi
    get_roi_images = mediapipeFaceMesh.get_roi_images(img, multi_face_landmarks)

    # # generate stacked roi images
    # roi_images = np.zeros_like(img)
    # if len(get_roi_images) > 0:
    #     roi_images = imageProcessing.stack_images(1, get_roi_images)

    # show fps
    fps = basicTool.countFps(my_time=time.time())
    cv2.putText(img_ori, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, global_color, 3)

    # show images in stacked
    stacked_images = imageProcessing.stack_images(1, ([img_ori, roi_images]))
    cv2.imshow("Stacked Image", stacked_images)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
