'''
This program is an improvement of hog_people_detection
It also detect face in the image, if this face
'''

import cv2
import numpy as np

def range_overlap(x1, x2, x3, x4):
    x_min = max(x1, x3)
    x_max = min(x2, x4)

    if x_min > x_max:
        return 0
    return x_max - x_min

def rec_overlap(rec1, rec2, percentage):
    r1_x1, r1_y1, r1_w, r1_h = rec1
    r1_x2, r1_y2 = r1_x1 + r1_w, r1_y1 + r1_h
    r1_area = r1_w * r1_h

    r2_x1, r2_y1, r2_w, r2_h = rec2
    r2_x2, r2_y2 = r2_x1 + r2_w, r2_y1 + r2_h
    r2_area = r2_w * r2_h

    x_range = range_overlap(r1_x1, r1_x2, r2_x1, r2_x2)
    y_range = range_overlap(r1_y1, r1_y2, r2_y1, r2_y2)


    return 1. * x_range * y_range / min(r1_area, r2_area) >= percentage

if __name__ == '__main__':
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')

    # camera = cv2.VideoCapture('./samples/768x576.mp4')
    camera = cv2.VideoCapture(0)

    while True:

        ret, frame = camera.read()
        frame = np.fliplr(frame).astype('uint8')

        recs, weights = hog.detectMultiScale(frame, winStride=(4, 4),
                                             padding=(8, 8), scale=1.05)
        hog_recs = []

        for rec in recs:
            for chosen in hog_recs:
                if rec_overlap(rec, chosen, 0.7):
                    break
            else:
                hog_recs.append(rec)

        recs = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

        face_recs = []

        for rec in recs:
            for hog_rec in hog_recs:
                if rec_overlap(rec, hog_rec, 0.7):
                    break
            else:
                face_recs.append(rec)

        for (x, y, w, h) in hog_recs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

        for (x, y, w, h) in face_recs:
            cv2.rectangle(frame, (x - w, y), (x + w * 2, y + h * 6), (0, 0, 255), 4)

        cv2.imshow('Human detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()