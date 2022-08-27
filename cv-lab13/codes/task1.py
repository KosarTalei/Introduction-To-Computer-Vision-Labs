""" Write a program that reads the images from the dataset and randomly displays four
examples of each Persian digit.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random


DATADIR = './digit_dataset/train/'
CATEGORIES = ['0', '1','2','3','4','5','6','7','8','9']#each Persian digit
IMG_SIZE = 200


for i in range(len(CATEGORIES)):
    
    training_data = []
    category = CATEGORIES[i]
    path = os.path.join(DATADIR, category)
        
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))#joins the paths
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array,category])
        
    #randomly
    random.shuffle(training_data)   
    #first 4
    for sample in training_data[:4]:
    
        cv2.imshow(sample[1],sample[0])
    
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    #if cv2.waitKey(0) & 0xFF == ord('q'):
        #break
