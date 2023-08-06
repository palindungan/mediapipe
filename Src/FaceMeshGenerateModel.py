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

folder = basicTool.get_base_url() + '/Resource/fotoPeserta/'
database = {}

for filename in listdir(folder):

    path = folder + filename
    gbr1 = cv2.imread(folder + filename)

    # processing
    multi_face_landmarks = mediapipeFaceMesh.processing(gbr1)
    get_roi_images = mediapipeFaceMesh.get_roi_images(gbr1, multi_face_landmarks)

    for roi_idx, roi_image in enumerate(get_roi_images):
        face = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))

        face = expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)

        database[os.path.splitext(filename)[0]] = signature

print(database)

myfile = open(basicTool.get_base_url() + '/Resource/' + 'data.pkl', "wb")
pickle.dump(database, myfile)
myfile.close()
