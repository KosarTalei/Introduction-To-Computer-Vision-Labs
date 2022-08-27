import cv2
import numpy as np
from matplotlib import pyplot as plt

fname = 'crayfish.jpg'
#fname = 'office.jpg'

I = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

f, axes = plt.subplots(2, 3)

# Transformation to obtain stretching
a=100
b=180
#a = I.min()
#b = I.max()
J = (I-a)*255.0 / (b-a)
J[J<0] = 0 #all element if<0
J[J> 255] = 255 #all element if>255
J=J.astype(np.uint8)

plt.xlabel('intensity')
plt.ylabel('number of pixels')

#part2
K = cv2.equalizeHist(I)

axes[0,1].imshow(J, 'gray', vmin=0, vmax=255)
axes[0,2].imshow(K, 'gray', vmin=0, vmax=255)
axes[0,0].imshow(I, 'gray', vmin=0, vmax=255)

axes[0,1].axis('off')
axes[0,2].axis('off')
axes[0,0].axis('off')

axes[1,0].hist(I.ravel(),256,[0,256]);
axes[1,1].hist(J.ravel(),256,[0,256]);
axes[1,2].hist(K.ravel(),256,[0,256]);

plt.show()

