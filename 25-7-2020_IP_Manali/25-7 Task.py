import cv2
import numpy as np

cap=cv2.VideoCapture(0)


lk_params = dict( winSize = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0=cv2.goodFeaturesToTrack(old_gray,10,0.2,1)

mask = np.zeros_like(old_frame)
while (1):
    ret,frame=cap.read()
    frame_gray= cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    pt1,st,err= cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,**lk_params)

    good_new = pt1[st == 1]
    good_old = p0[st == 1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b=new.ravel()
        c,d=old.ravel()
        if a-c >0:
            print(a,c)
            print("Actual-RIGHT")
            print("Mirror- LEFT")
        if a-c <0:
            print(a,c)
            print("Actual- LEFT")
            print("Mirror-RIGHT")
        else:
            print(a,c)
            print("No Movement")
    img= cv2.add(frame, mask)

    img=cv2.circle(img,(c,d),1,(255,255,255),-1)

    cv2.imshow('frame',img)

    k = cv2.waitKey(1)
    if k == 32:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()

