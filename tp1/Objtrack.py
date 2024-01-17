import cv2
import numpy as np
import matplotlib.pyplot as plt
from Kalman import KalmanFilter
from Detector import detect

Karman = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)

video = cv2.VideoCapture("data/tp1/randomball.avi")

predicted = []
updated = []

while True:
    ret, frame = video.read()
    if frame is None:
        break
    center = detect(frame)
    if len(center) > 0:
        state = Karman.predict()
        rec_x = state[0][0]
        rec_y = state[1][0]
        predicted.append([rec_x, rec_y])
        cv2.rectangle(frame, (int(rec_x - 15), int(rec_y - 15)), (int(rec_x + 15), int(rec_y + 15)), (255, 0, 0), 2)


        state = Karman.update(center[0])
        rec_x = state[0][0]
        rec_y = state[1][0]
        updated.append([rec_x, rec_y])
        for i in range(len(updated)-1):
            cv2.line(frame, (int(updated[i][0]), int(updated[i][1])), (int(updated[i+1][0]), int(updated[i+1][1])), (0, 0, 255), 2)
        

        cv2.rectangle(frame, (int(rec_x - 15), int(rec_y - 15)), (int(rec_x + 15), int(rec_y + 15)), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
