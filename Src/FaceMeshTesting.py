import cv2
import time

from Util import BasicToolModule
from Util import ImageProcessingModule
from Util import MediapipeFaceMeshModule

# Utility Class
basicTool = BasicToolModule.BasicTool()
imageProcessing = ImageProcessingModule.ImageProcessing()
mediapipeFaceMesh = MediapipeFaceMeshModule.MediapipeFaceMesh()

# Camera Setting
w_cam, h_cam = 480, 360  # width and height image
no_cam = 1  # default Cam
camera_brightness = 190  # set brightness
global_color = (0, 255, 0)  # default color

cap = cv2.VideoCapture(basicTool.get_base_url() + "/Resource/Videos/3.mp4")  # read file

# # Webcam Video
# cap = cv2.VideoCapture(no_cam)  # webcam
# cap.set(3, w_cam)  # width
# cap.set(4, h_cam)  # height
# cap.set(10, camera_brightness)  # brightness

while True:
    success, img = cap.read()
    img_ori = img.copy()

    # processing
    multi_face_landmarks = mediapipeFaceMesh.processing(img)
    img = mediapipeFaceMesh.drawing_roi(img, multi_face_landmarks)

    # show fps
    fps = basicTool.countFps(my_time=time.time())
    cv2.putText(img_ori, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, global_color, 3)

    # show images in stacked
    stacked_images = imageProcessing.stack_images(1, ([img_ori, img]))
    cv2.imshow("Stacked Image", stacked_images)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
