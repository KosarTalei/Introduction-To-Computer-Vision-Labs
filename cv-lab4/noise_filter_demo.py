import numpy as np
import cv2

I = cv2.imread('branches2.jpg').astype(np.float64) / 255

J1 = I.astype(np.float32) #only 8u and 32f images for bilateral filter

noise_sigma = 0.04  # initial standard deviation of noise

m = 1  # initial filter size,

gm = 3  # gaussian filter size

size = 9  # bilateral filter size
sigmaColor = 0.3
sigmaSpace = 75

# with m = 1 the input image will not change
filter = 'b'  # box filter

while True:

    # add noise to image
    N = np.random.rand(*I.shape) * noise_sigma
    N=N.astype(np.float32)
    J = I + N

    #N1 = np.random.rand(*J1.shape) * noise_sigma

    if filter == 'b':
        # filter with a box filter
        m=6
        K = cv2.boxFilter(J,-1,(m,m))
    elif filter == 'g':
        # filter with a Gaussian filter
        m=11
        K = cv2.GaussianBlur(J,(gm,gm),0)
    elif filter == 'l':
        # filter with a bilateral filter
        K = cv2.bilateralFilter(J,size, sigmaColor, sigmaSpace)

    # filtered image

    cv2.imshow('img', K)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('b'):
        filter = 'b'  # box filter
        print('Box filter')

    elif key == ord('g'):
        filter = 'g'  # filter with a Gaussian filter
        print('Gaussian filter')

    elif key == ord('l'):
        filter = 'l'  # filter with a bilateral filter
        print('Bilateral filter')

    elif key == ord('+'):
        # increase m
        m = m + 2
        print('m=', m)
    elif key == ord('-'):
        # decrease filter size m
        if m >= 3:
            m = m - 2
        print('m=', m)
    elif key == ord('u'):
        # increase noise
        noise_sigma += 0.02
        if(noise_sigma > 1):
            noise_sigma = 1.0
    elif key == ord('d'):
        # decrease noise
        noise_sigma -= 0.02
        if(noise_sigma < 0):
            noise_sigma = 0.0
    elif key == ord('p'):
        # increase gm
        gm = gm + 2 #odd
        #print('gm=', gm)
    elif key == ord('n'):
        # decrease gm
        gm = gm - 2
        if gm < 0:
            gm = 1
        #sigmaColor -= 0.02
        #if sigmaColor < 0:
            #sigmaColor = 0
            
    elif key == ord('>'):
        # increase size
        # filter with a bilateral filter
        size = size + 3
        #print('size=', size)
            
    elif key == ord('<'):
        # decrease size
        # filter with a bilateral filter
         size = size - 3
         if size <-1:
            size = -1 #computed from Sigma Space
    elif key == ord('q'):
        break  # quit

cv2.destroyAllWindows()
