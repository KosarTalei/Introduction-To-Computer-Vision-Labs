import numpy as np
import cv2

I = cv2.imread('beans.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

#sigma = 15
#G = cv2.GaussianBlur(G, (sigma,sigma), 0); # blur the image
#ret, T = cv2.threshold(G, 125, 255, cv2.THRESH_BINARY)
ret, T = cv2.threshold(G, 127, 255, cv2.THRESH_BINARY)

4
cv2.imshow('Thresholded', T)
cv2.waitKey(0) # press any key to continue...

## erosion 
kernel = np.ones((19,19),np.uint8)# change 5 to 19
T = cv2.erode(T,kernel)
cv2.imshow('After Erosion', T)
cv2.waitKey(0) # press any key to continue...

n,C = cv2.connectedComponents(T);
#n = np.max(C)

font = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(T,'There are %d beans!'%(n-1),(20,40), font, 1, 255,2)
cv2.imshow('Num', T)
cv2.waitKey(0)
