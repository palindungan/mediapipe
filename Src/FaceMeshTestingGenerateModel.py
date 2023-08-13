import os
from os import listdir

import pickle
import cv2

from keras_facenet import FaceNet

from Util import BasicToolModule

# utility class
basicTool = BasicToolModule.BasicTool()

MyFaceNet = FaceNet()

folder = basicTool.get_base_url() + '/Resource/Dataset/'
database = {}

for label in listdir(folder):
    file_path = folder + label + '/'
    faces = []

    for file_name in listdir(file_path):
        img = cv2.imread(file_path + file_name)

        face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))
        faces.append(face)

    if len(faces) > 0:
        signature = MyFaceNet.embeddings(faces)
        database[os.path.splitext(label)[0]] = signature

print(database)

file = open(basicTool.get_base_url() + '/Resource/' + 'database.pkl', "wb")
pickle.dump(database, file)
file.close()
