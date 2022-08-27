import cv2
import numpy as np
#from matplotlib import pyplot as plt

I = cv2.imread('masoleh.jpg')

## your code ##


# notice that OpenCV uses BGR instead of RGB!

cv2.imshow('win1',I)
#new matrix of image with same size as I and set all elements to 0.
img = np.zeros(I.shape, dtype=np.uint8) 

while 1:
    k = cv2.waitKey()

    if k == ord('o'):
        cv2.imshow('win1',I)
    elif k == ord('b'):
        B = img.copy()  #a copy of an image:
        B[:, :, 0] = I[:, :, 0] #only have blue channel of I, all the G & R set to 0.
        cv2.imshow('win1',B)
    elif k == ord('g'):
        G = img.copy()
        G[:, :, 1] = I[:, :, 1]
        cv2.imshow('win1',G)
    elif k == ord('r'):
        R = img.copy()
        R[:, :, 2] = I[:, :, 2]
        cv2.imshow('win1',R)
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break
    
 
    

