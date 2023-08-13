import cv2
import time

from Util import BasicToolModule
from Util import ImageProcessingModule
from Util import MediapipeFaceMeshModule
from Util import FaceRecognitionModule

# utility class
basicTool = BasicToolModule.BasicTool()
imageProcessing = ImageProcessingModule.ImageProcessing()
mediapipeFaceMesh = MediapipeFaceMeshModule.MediapipeFaceMesh()
faceRecognition = FaceRecognitionModule.FaceRecognition()

# camera setting
w_cam, h_cam = 480, 360  # width and height image
no_cam = 1  # default Cam
camera_brightness = 190  # set brightness
global_color = (0, 255, 0)  # default color

# cap = cv2.VideoCapture(basicTool.get_base_url() + "/Resource/Videos/2.mp4")  # read file

# webcam
cap = cv2.VideoCapture(no_cam)  # webcam
cap.set(3, w_cam)  # width
cap.set(4, h_cam)  # height
cap.set(10, camera_brightness)  # brightness

while True:
    success, img = cap.read()
    img_ori = img.copy()

    # MediapipeFaceMesh processing
    multi_face_landmarks = mediapipeFaceMesh.processing(img)
    get_roi_images = mediapipeFaceMesh.get_roi_images(img, multi_face_landmarks)
    roi_images, roi_bboxes = get_roi_images

    # MAIN LOGIC
    for roi_idx, roi_image in enumerate(roi_images):
        roi_bbox = roi_bboxes[roi_idx]

        identity = faceRecognition.prediction(roi_image)  # FaceRecognition

        cv2.rectangle(img_ori, (roi_bbox[0], roi_bbox[1]), (roi_bbox[2], roi_bbox[3]), global_color, 2)
        cv2.putText(img_ori, identity, (roi_bbox[0], roi_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, global_color, 2,
                    cv2.LINE_AA)

    # show fps
    fps = basicTool.count_fps(my_time=time.time())
    cv2.putText(img_ori, f'FPS {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, global_color, 3)

    # show images in stacked
    stacked_images = imageProcessing.stack_images(1, ([img_ori]))
    cv2.imshow("Stacked Image", stacked_images)

    # action for end proses
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
