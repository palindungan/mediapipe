import os
from os import listdir
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import load_model
import numpy as np

import pickle
import cv2

from keras_facenet import FaceNet

# ------------------

import time

from Util import BasicToolModule
from Util import ImageProcessingModule
from Util import MediapipeFaceMeshModule

# utility class
basicTool = BasicToolModule.BasicTool()
imageProcessing = ImageProcessingModule.ImageProcessing()
mediapipeFaceMesh = MediapipeFaceMeshModule.MediapipeFaceMesh()

MyFaceNet = FaceNet()

folder = basicTool.get_base_url() + '/Resource/Dataset/'
database = {}

for label in listdir(folder):
    file_path = folder + label + '/'
    faces = []

    for file_name in listdir(file_path):
        img = cv2.imread(file_path + file_name)

        # processing
        multi_face_landmarks = mediapipeFaceMesh.processing(img)
        get_roi_images = mediapipeFaceMesh.get_roi_images(img, multi_face_landmarks)
        roi_images, roi_bboxes = get_roi_images

        for roi_idx, roi_image in enumerate(roi_images):
            face = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (160, 160))
            faces.append(face)

            # cv2.imshow("roi_image", roi_image)
            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     break

    if len(faces) > 0:
        signature = MyFaceNet.embeddings(faces)
        database[os.path.splitext(label)[0]] = signature

print(database)

myfile = open(basicTool.get_base_url() + '/Resource/' + 'data_rizki.pkl', "wb")
pickle.dump(database, myfile)
myfile.close()
