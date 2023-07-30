import sys
import os
import numpy as np
import cv2
import time


class BasicTool():
    def __init__(self):
        self.pTime = 0
        self.countFolder = 0
        self.count = 0
        self.countSave = 0

    def countFps(self, time):
        cTime = time
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        return fps

    def getBaseUrl(self):
        return sys.path[1]

    def empty(self, a):
        pass

    def CreateDirectory(self, myPath):
        self.countFolder = 0

        while os.path.exists(myPath + str(self.countFolder)):
            self.countFolder = self.countFolder + 1

        os.makedirs(myPath + str(self.countFolder))

    def CreateBlankImage(self, img):
        width = img.shape[1]
        height = img.shape[0]
        return np.zeros((height, width, 3), np.uint8)

    def saveImageRoi(self, img, moduleVal, minBlur, myPath):
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if self.count % moduleVal == 0 and blur > minBlur:
            nowTime = time.time()
            cv2.imwrite(
                myPath +
                str(self.countFolder) +
                '/' +
                str(self.countSave) +
                '_' +
                str(int(blur)) +
                '_' +
                str(nowTime) + '.png',
                img)
            self.countSave = self.countSave + 1
        self.count = self.count + 1

        return self.countSave
