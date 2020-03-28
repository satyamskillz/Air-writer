from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
import argparse

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
yellowLower = (20, 100, 100)
yellowUpper = (30, 255, 255)
pts = deque(maxlen=50)

cv = cv2.VideoCapture(0)
first = None
stop = 0
while True:
    ret, frame = cv.read()
    frame = imutils.resize(frame, width=600)
    if (stop == 10):
        first = frame
    stop += 1
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, greenLower, greenUpper)  # for green color
    mask2 = cv2.inRange(hsv, yellowLower, yellowUpper)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=6)
    maskOpp = cv2.bitwise_not(mask)

    if (first is None):
        continue
    else:
        image = cv2.bitwise_and(first, first, mask=maskOpp)
    #         cv2.imshow("initial frame",image)

    video = cv2.bitwise_and(frame, frame, mask=mask)

    img = cv2.add(image, video)
    #     img1=cv2.addWeighted(image,1,video,1,1)
    cv2.imshow("object",img)

    #     findind contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if (len(cnts) > 0):
        c = max(cnts, key=cv2.contourArea)  # finding maximum area
        ((x, y), radius) = cv2.minEnclosingCircle(c)  # circle property extraction
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if (radius > 2):
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if (pts[i - 1] is None or pts[i] is None):
            continue

        thickness = int(np.sqrt(50 / float(i + 1)) * 2)
        cv2.line(img, pts[i - 1], pts[i], (0, 255, 0), thickness)

    cv2.imshow("video", img)
    cv2.imshow("original", frame)
    c = cv2.waitKey(1)
    if (c == 27):
        break
cv.release()
cv2.destroyAllWindows()