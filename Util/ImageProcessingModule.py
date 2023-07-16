# Start of import library
import cv2
import numpy as np


class ImageProcessing():

    def stackImages(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver

    def getContours(self, img, imgContour, trackArea):
        # find contours
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # draw contours
        for contour in contours:
            area = cv2.contourArea(contour)  # get area contour
            if area >= trackArea:

                # draw contour
                color = (0, 0, 255)
                thickness = 7
                cv2.drawContours(imgContour, contour, -1, color, thickness)

                # detect approx poly DP
                epsilon = 0.02 * cv2.arcLength(contour, True)  # maximum distance from contour to approximated contour.
                approx = cv2.approxPolyDP(contour, epsilon, True)  # approximates a contour shape to another shape

                # draw circle each sharp edge
                for i in range(0, len(approx)):
                    cv2.circle(imgContour, (approx[i][0][0], approx[i][0][1]), 20, (0, 255, 0), cv2.FILLED)

                # draw rectangle
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (255, 0, 0), 5)

                # create text
                cv2.putText(imgContour, "Point: " + str(len(approx)), (x + w + 20, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255), 2)
                cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255), 2)

    def preProcessing(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to GRAY
        img = cv2.equalizeHist(img)  # function to balance the histogram (contras effect)
        img = img / 255  # from 0,1,2,3 ... 254,255 to 0,1
        return img
