import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

old_x, old_y = 0, 0

while True:
    ret, image = cap.read(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 200, 0), 2)
        if (old_x > x):
            dir = 'YOU MOVED LEFT'
        elif (old_x < x):
            dir = 'YOU MOVED RIGHT'
        else:
            dir = 'YOU DID NOT MOVED'
        old_x, old_y = x, y
        cv2.putText(image, "DIRECTION: " + dir, (150, 400), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    new_image = cv2.resize(image, (600, 600))
    cv2.imshow("Direction", new_image)

    k = cv2.waitKey(1)
    if (k == 32):
        break

cap.release()
cv2.destroyAllWindows()