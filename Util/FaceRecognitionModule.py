import cv2
import numpy as np
import pickle

from keras_facenet import FaceNet

from Util import BasicToolModule


class FaceRecognition:
    def __init__(self):
        self.face_net = FaceNet()

        self.basicTool = BasicToolModule.BasicTool()

        model_file = open(self.basicTool.get_base_url() + '/Resource/' + 'database_testing.pkl', "rb")
        self.model_database = pickle.load(model_file)
        model_file.close()

    def prediction(self, roi_image):
        face = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))

        face = np.expand_dims(face, axis=0)
        signature = self.face_net.embeddings(face)

        min_dist = 6
        identity = 'unknown'
        for key, value in self.model_database.items():
            dist = np.linalg.norm(value - signature)
            print(dist)
            if dist < min_dist:
                min_dist = dist
                identity = key

        return identity
