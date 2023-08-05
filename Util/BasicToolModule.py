import sys
import os
import numpy as np
import cv2
import time


class BasicTool:
    def __init__(self):
        self.p_time = 0
        self.count = 0
        self.count_folder = 0
        self.count_saved = 0

    def countFps(self, my_time):
        c_time = my_time
        fps = 1 / (c_time - self.p_time)
        self.p_time = c_time

        return fps

    @staticmethod
    def get_base_url():
        return sys.path[1]

    def save_image(self, path, img, module_val, min_blur):
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if self.count % module_val == 0 and blur > min_blur:
            now = time.time()
            cv2.imwrite(
                path +
                str(self.count_folder) +
                '/' +
                str(self.count_saved) +
                '_' +
                str(int(blur)) +
                '_' +
                str(now) + '.png',
                img
            )
            self.count_saved = self.count_saved + 1
        self.count = self.count + 1

        return self.count_saved
