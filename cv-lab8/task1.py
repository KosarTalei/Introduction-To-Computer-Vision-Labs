import cv2
import numpy as np

I = cv2.imread('polygons.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

ret, T = cv2.threshold(G,220,255,cv2.THRESH_BINARY_INV)

nc1,CC1 = cv2.connectedComponents(T)

for k in range(1,nc1):

    Ck = np.zeros(T.shape, dtype=np.float32)
    Ck[CC1 == k] = 1;
    Ck = cv2.GaussianBlur(Ck,(5,5),0)
    #Ck = cv2.cvtColor(Ck,cv2.COLOR_GRAY2BGR)

    # Now, apply corner detection on Ck
    window_size = 5#3
    soble_kernel_size  = 3# kernel size for gradients
    alpha = 0.04
    Ck = cv2.cornerHarris(Ck,window_size,soble_kernel_size,alpha)
    # normalize C so that the maximum value is 1
    Ck = Ck/ Ck.max()

    Ck = np.uint8(Ck > 0.01) * 255


    # plot centroids of connected components as corner locations
    nc2,CC2, stats, centroids = cv2.connectedComponentsWithStats(Ck)
    J = I.copy()
    for i in range(1,nc2):
        cv2.circle(J,(int(centroids[i,0]),int(centroids[i,1])) ,3,(0,0,255))


    #nc2,CC2 = cv2.connectedComponents(Ck)
    nc2 = nc2-1
    
    Ck = cv2.cvtColor(Ck,cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(Ck,'There are %d vertices!'%(nc2),(20,30), font, 1,(0,0,255),1)

    
    cv2.imshow('corners',Ck)
    cv2.waitKey(0) # press any key



