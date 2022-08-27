import numpy as np
import cv2

# create a VideoCapture object
cap = cv2.VideoCapture('eggs.avi')

# get the dimensions of the frame
# you can also read the first frame to get these
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width of the frame
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # height of the frame

fourcc = cv2.VideoWriter_fourcc(*'XVID') # choose codec

# create VideoWriter object w by h, 30 frames per second
out = cv2.VideoWriter('eggs-reverse.avi',fourcc, 30.0, (w,h))

frame_buffer = []

while True:
    ret, I = cap.read()
    frame_buffer.append(I)
    
    if ret == False: # end of video (or error)
        break
    
frame_buffer.pop() #end of video,none

#reverce frame buffer
frame_buffer.reverse()
 
for frame in frame_buffer:
    cv2.imshow("eggs-reverse" , frame)
    out.write(frame)

    key = cv2.waitKey(30) # ~ 30 frames per second

    if key & 0xFF == ord('q'): 
        break
    
cv2.destroyAllWindows()

cap.release()
out.release()
