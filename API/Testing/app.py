import numpy as np
import cv2

from flask import *
from Util import BasicToolModule
from Util import ImageProcessingModule
from Util import MediapipeFaceMeshModule
from Util import FaceRecognitionModule
from Util import AntiSpoofingModule

app = Flask(__name__)

# utility class
basicTool = BasicToolModule.BasicTool()
imageProcessing = ImageProcessingModule.ImageProcessing()
mediapipeFaceMesh = MediapipeFaceMeshModule.MediapipeFaceMesh()
faceRecognition = FaceRecognitionModule.FaceRecognition()
antiSpoofing = AntiSpoofingModule.AntiSpoofing()


@app.route('/')
def main():
    return "index.html"


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        data = []

        file = request.files['file']

        # Convert the image data into a NumPy array
        image_data = cv2.imread(basicTool.get_base_url() + "/Resource/Images/ktp.jpg")

        # Decode the image using OpenCV
        img = image_data
        img_ori = img.copy()

        img_height, img_width, img_channel = img.shape

        # data_request = request.form

        # MediapipeFaceMesh processing
        multi_face_landmarks = mediapipeFaceMesh.processing(img)
        get_roi_images = mediapipeFaceMesh.get_roi_images(img, multi_face_landmarks)
        roi_images, roi_bboxes = get_roi_images

        # MAIN LOGIC
        for roi_idx, roi_image in enumerate(roi_images):
            roi_bbox = roi_bboxes[roi_idx]

            spoof_x_diff = abs(int(roi_bbox[2] - roi_bbox[0]))
            spoof_x_diff = int(spoof_x_diff / 2)
            spoofing_bbox = [
                int(roi_bbox[0] - spoof_x_diff), 0,
                int(roi_bbox[2] + spoof_x_diff), img_height
            ]
            spoof_img_cropped = img_ori[
                                abs(spoofing_bbox[1]): abs(spoofing_bbox[3]),
                                abs(spoofing_bbox[0]):abs(spoofing_bbox[2])
                                ]
            # cv2.imshow("spoof_img_cropped", spoof_img_cropped)
            model_dir = basicTool.get_base_url() + "/Resource/AntiSpoof/" + "resources/anti_spoof_models"
            label, value = antiSpoofing.test(spoof_img_cropped,
                                             model_dir,
                                             0)
            if label == 1 and value >= 0.95:
                identity = faceRecognition.prediction(roi_image)  # FaceRecognition
            else:
                identity = "FAKE"

            data.append({
                "identity": identity,
                "score": value * 100,
                "threshold": 95,
            })

        return jsonify({'success': True, 'message': 'success', 'data': data}), 201


if __name__ == '__main__':
    app.run(debug=True)
