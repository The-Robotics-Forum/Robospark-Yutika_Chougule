import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret , old = cap.read(0)
prvs = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(old)
hsv[...,1] = 255

while True:
    ret, new = cap.read(0)
    next = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("original", rgb)
    old=new
    k=cv2.waitKey(1)
    if k==32:
        break

cap.release()
cv2.destroyAllWindows()
