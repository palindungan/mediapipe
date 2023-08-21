import numpy as np
import cv2

from flask import *
from Util import MediapipeFaceMeshModule

app = Flask(__name__)
mediapipeFaceMesh = MediapipeFaceMeshModule.MediapipeFaceMesh()


@app.route('/')
def main():
    return "index.html"


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        file = request.files['file']

        # Convert the image data into a NumPy array
        image_data = np.frombuffer(file.read(), np.uint8)

        # Decode the image using OpenCV
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        data = request.form

        # MediapipeFaceMesh processing
        multi_face_landmarks = mediapipeFaceMesh.processing(img)
        get_roi_images = mediapipeFaceMesh.get_roi_images(img, multi_face_landmarks)
        roi_images, roi_bboxes = get_roi_images

        return {'data': data}, 201


if __name__ == '__main__':
    app.run(debug=True)
