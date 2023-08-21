import os
import cv2
import numpy as np
import warnings
import time

from Util import BasicToolModule

from Util.AntiSpoof.anti_spoof_predict import AntiSpoofPredict
from Util.AntiSpoof.generate_patches import CropImage
from Util.AntiSpoof.utility import parse_model_name

warnings.filterwarnings('ignore')

basicTool = BasicToolModule.BasicTool()

SAMPLE_IMAGE_PATH = basicTool.get_base_url() + "/Resource/AntiSpoof/" + "images/sample/"


class AntiSpoofing:
    def __init__(self):
        self.basicTool = BasicToolModule.BasicTool()

    @staticmethod
    def check_image(image):
        height, width, channel = image.shape
        if width / height != 3 / 4:
            print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
            return False
        else:
            return True

    def test(self, image, model_dir, device_id):
        model_test = AntiSpoofPredict(device_id)
        image_cropper = CropImage()
        # image = cv2.resize(image, (int(image.shape[0] * 3 / 4), image.shape[0]))
        # result = self.check_image(image)
        # if result is False:
        #     return
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time() - start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label] / 2

        if label == 1 and value > 0.9:
            print("Image '{}' is Real. Score: {:.2f}.".format('image_name', value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake. Score: {:.2f}.".format('image_name', value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))

        return label, value
