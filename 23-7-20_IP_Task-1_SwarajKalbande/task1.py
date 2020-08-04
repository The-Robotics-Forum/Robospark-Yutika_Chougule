import cv2

print(cv2.__version__)

img = cv2.imread(r"C:\Users\swara\OneDrive\Desktop\Day 1\Task Image\text.jpeg", cv2.IMREAD_UNCHANGED)
img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img3=cv2.resize(img, (600,600))

cv2.imshow("color", img)
cv2.imshow("greyscale", img1)
cv2.imshow("hsv", img2)
cv2.imshow("rrsize",img3)

print(img.shape)
print(img3.shape)
print(img.size)
print(img3.size)

ret, threshold =cv2.threshold(img, 150, 255,cv2.THRESH_BINARY)
ret, thresh2 =cv2.threshold(img, 150, 255,cv2.THRESH_BINARY_INV)
ret, thresh1 =cv2.threshold(img1, 150, 255,cv2.THRESH_OTSU)


cv2.imshow('threshold', threshold)
cv2.imshow('thresholdotsu',thresh1)
cv2.imshow('threshold_inv',thresh2)



cv2.waitKey(0)
cv2.destroyAllWindows()