import numpy as np
import cv2

I = cv2.imread('isfahan.jpg', cv2.IMREAD_GRAYSCALE);
I = I.astype(float) / 255

sigma = 0.04 # initial standard deviation of noise 

while True:
    N = np.random.randn(*I.shape) * sigma
    
    J = I+N; # change this line so J is the noisy image
    
    cv2.imshow('snow noise',J)

    
    # press any key to exit
    key = cv2.waitKey(33)
    if key & 0xFF == ord('u'): # increase noise
        sigma += 0.02
        if(sigma > 1):
            sigma = 1.0
    elif key & 0xFF == ord('d'):  # decrease noise
        sigma -= 0.02
        if(sigma < 0):
            sigma = 0.0
    elif key & 0xFF == ord('q'): #quit
        break
    
cv2.destroyAllWindows()
