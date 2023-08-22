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

        # data_request = request.form
        file = request.files['file']
        image_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # MediapipeFaceMesh processing
        multi_face_landmarks = mediapipeFaceMesh.processing(img)
        get_roi_images = mediapipeFaceMesh.get_roi_images(img, multi_face_landmarks)
        roi_images, roi_bboxes = get_roi_images

        # MAIN LOGIC
        for roi_idx, roi_image in enumerate(roi_images):
            identity, dist = faceRecognition.predictionV2(roi_image)  # FaceRecognition

            data.append({
                "identity": identity,
                "score": dist,
                "threshold": 6,
            })

        return jsonify({'success': True, 'message': 'success', 'data': data}), 201


@app.route('/anti-spoofing', methods=['POST'])
def antiSpoofing():
    if request.method == 'POST':
        data = []

        # data_request = request.form
        file = request.files['file']
        image_data = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        img_ori = img.copy()

        model_dir = basicTool.get_base_url() + "/Resource/AntiSpoof/" + "resources/anti_spoof_models"
        label, value = antiSpoofing.test(img_ori, model_dir, 0)
        if label == 1 and value >= 0.95:
            identity = "Real"
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
