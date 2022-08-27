import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
color = (255.0)
fontScale = 0.3
thickness = 1

def std_filter(I, ksize):
    F = np.ones((ksize,ksize), dtype=float) / (ksize*ksize);
       
    MI = cv2.filter2D(I,-1,F) # apply mean filter on I

    I2 = I * I; # I squared
    MI2 = cv2.filter2D(I2,-1,F) # apply mean filter on I2

    return np.sqrt(MI2 - MI * MI)

def zero_crossing(I):
    """Finds locations at which zero-crossing occurs, used for
    Laplacian edge detector"""
    
    Ishrx = I.copy();
    Ishrx[:,1:] = Ishrx[:,:-1]
        
    Ishdy = I.copy();
    Ishdy[1:,:] = Ishdy[:-1,:]
        
    ZC = (I==0) | (I * Ishrx < 0) | (I * Ishdy < 0); # zero crossing locations

    SI = std_filter(I, 3) / I.max()

    Mask =  ZC & (SI > .1)

    E = Mask.astype(np.uint8) * 255 # the edges

    return E

cam_id = 0  # camera id

# for default webcam, cam_id is usually 0
# try out other numbers (1,2,..) if this does not work

cap = cv2.VideoCapture(cam_id)

mode = 'o' # show the original image at the beginning

sigma = 5

while True:
    ret, I = cap.read();
    #I = cv2.imread("agha-bozorg.jpg") # can use this for testing 
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) # convert to grayscale
    Ib = cv2.GaussianBlur(I, (sigma,sigma), 0); # blur the image
    
    if mode == 'o':
        # J = the original image
        J = I
    elif mode == 'x':
        # J = Sobel gradient in x direction
        J = np.abs(cv2.Sobel(Ib,cv2.CV_64F,1,0));
        J = cv2.putText(J, 'Sobel gradient in x direction', org, font, fontScale, color,thickness,cv2.LINE_AA)
    elif mode == 'y':
        # J = Sobel gradient in y direction
        J = np.abs(cv2.Sobel(Ib,cv2.CV_64F,0,1))
        J = cv2.putText(J, 'Sobel gradient in y direction', org, font,fontScale,color,thickness,cv2.LINE_AA)
    
    elif mode == 'm':
        # J = magnitude of Sobel gradient
        JX = np.abs(cv2.Sobel(Ib,cv2.CV_64F,1,0));
        JY = np.abs(cv2.Sobel(Ib,cv2.CV_64F,0,1))
        J = np.sqrt(JX*JX + JY*JY)
        J = cv2.putText(J, 'Magnitude of Sobel gradient', org, font,fontScale,color,thickness,cv2.LINE_AA)
    
    elif mode == 's':
        # J = Sobel + thresholding edge detection
        thresh = 90 # threshold
        Jx = cv2.Sobel(Ib,cv2.CV_64F,1,0)
        Jy = cv2.Sobel(Ib,cv2.CV_64F,0,1)
        J = np.sqrt(Jx*Jx + Jy*Jy) 
        J= np.uint8(J > thresh)*255 # threshold the gradients
        J = cv2.putText(J, 'Sobel gradients + thresholding edge detection', org, font,fontScale,color,thickness,cv2.LINE_AA)
    
    elif mode == 'l':
        # J = Laplacian edges
        J = cv2.Laplacian(Ib,cv2.CV_64F,ksize=5)
        J = zero_crossing(J);
        J = cv2.putText(J, 'Laplacian edges', org, font,fontScale,color,thickness,cv2.LINE_AA)
    
    elif mode == 'c':
        # J = Canny edges
        lth = 50# low threshold
        hth = 120 # high threshold
        J = cv2.Canny(Ib,lth, hth)
        J = cv2.putText(J, 'Canny edges', org, font,fontScale,color,thickness,cv2.LINE_AA)
    # we set the image type to float and the
    # maximum value to 1 (for a better illustration)
    # notice that imshow in opencv does not automatically
    # map the min and max values to black and white. 
    J = J.astype(float) / J.max();
    cv2.imshow("my stream", J);

    key = chr(cv2.waitKey(1) & 0xFF)

    if key in ['o', 'x', 'y', 'm', 's', 'c', 'l']:
        mode = key
    if key == '-' and sigma > 1:
        sigma -= 2
        print("sigma = %d"%sigma)
    if key in ['+','=']:
        sigma += 2    
        print("sigma = %d"%sigma)
    elif key == 'q':
        break

cap.release()
cv2.destroyAllWindows()
