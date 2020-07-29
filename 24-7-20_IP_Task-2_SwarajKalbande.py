import cv2
import numpy as np

img = cv2.imread(r"C:\Users\swara\OneDrive\Desktop\Day 2\Task Image\rose_flower.jpg")
img2 = cv2.imread(r"C:\Users\swara\OneDrive\Desktop\Day 2\Task Image\pot.jpg")
res = cv2.resize(img, (300, 500))
pot = cv2.resize(img2,(300 , 500))


bgrl = np.array([90, 10, 120], np.uint8)
bgrh = np.array([190, 255, 255], np.uint8)

mask = cv2.inRange(res, bgrl, bgrh)

kernel = np.array(([0,0,0],[0,1,0],[0,0,0]), np.uint8)

blurred = cv2.filter2D(mask, -1, kernel)

avg = cv2.blur(mask, (5, 5))

gaus_blur = cv2.GaussianBlur(mask, (5, 5), 0)

median = cv2.medianBlur(mask, 7)


bilateral = cv2.bilateralFilter(res, 5, 111, 111)

image= cv2.addWeighted( res ,0.7 , pot, 0.3, 50)

font=cv2.FONT_ITALIC

rose=cv2.putText(res, 'ROSE',(100,250), font,2, (0,255,255), 3, cv2.LINE_AA)

cv2.imshow("rose", rose)
cv2.imshow("img", res)
cv2.imshow("masked original", mask)
cv2.imshow("blurred", blurred)
cv2.imshow("averaging", avg)
cv2.imshow("gaussian blur", gaus_blur)
cv2.imshow("median blur normal", median)
cv2.imshow("bilateral", bilateral)



cv2.imshow("final", image)


cv2.waitKey(0)
cv2.destroyAllWindows()
