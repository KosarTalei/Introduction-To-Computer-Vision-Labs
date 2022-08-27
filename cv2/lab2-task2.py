import cv2
import numpy as np

I = cv2.imread('damavand.jpg')
J = cv2.imread('eram.jpg')
while 1:
    
    for weight in np.arange(0,10):
        blend = weight/10.0
        K = cv2.addWeighted( I, 1-blend, J, blend, 0)
        cv2.imshow("Blending", K)
        cv2.waitKey(1000)

    if cv2.waitKey(2000) == ord('q'):
        cv2.destroyAllWindows()
        break  
