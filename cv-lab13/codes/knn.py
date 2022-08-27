""" Run the previous code (task2.py) using the KNN classifier:
"""

""" Run the previous code (task2.py) which uses raw pixels as input features and one vs. one
approach. Replace your classifier and its setting by the following:
(Note that the “rbf” is the default kernel of the SVC class.)
"""
""" The above accuracy is calculated on the train data (classification.py). This is not a good way to evaluate a
classifier, because it can overfit the training data, that is it can work very well on the
training data, but perform poorly on the new (unseen) data. Your task is to examine the
accuracy of the classifier both on the train and the test data.
"""


import random
from sklearn.preprocessing import StandardScaler
from concurrent import futures
import cv2
from functools import partial
from sklearn.svm import SVC
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier


def processing(dir, images_list, idx, file):
    print("Processing: " + file)
    temp = []

    for addr in images_list[idx]:
        I = cv2.imread(os.path.join(dir, file, addr))
        temp.append(I.ravel())

    return temp

# Don't bother yourself with this function
def extract_features(dir, images_list, files):
    data = []
    with futures.ProcessPoolExecutor() as executor:
        indices = np.arange(len(images_list))
        func = partial(processing, dir, images_list)
        results = executor.map(func, indices, files)

        for result in results:
            data.extend(result)
    return data


def main():
    train_dir = './digit_dataset/train/'
    train_labels = []
    train_images_list = []

    files = os.listdir(train_dir)
    files.sort()

    for idx, file in enumerate(files):
        images_list = os.listdir(train_dir + file)
        images_list.sort()
        train_labels.extend([idx] * len(os.listdir(train_dir + file)))
        train_images_list.append(images_list)

    print("------------Feature extraction for train data set------------")
    train_data = extract_features(dir=train_dir, images_list=train_images_list, files=files)
    print("------------End of extraction--------------------------------")


    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)

    # Making the KNN Classifer
    
    classifier = KNeighborsClassifier(n_neighbors=6)
    # Training the model on the training data and labels
    classifier.fit(train_data, train_labels)

    test_dir = './digit_dataset/test/'
    files = os.listdir(test_dir)
    files.sort()

    test_labels = []
    test_images_list = []

    for idx, file in enumerate(files):
        images_list = os.listdir(test_dir + file)
        images_list.sort()
        test_labels.extend([idx] * len(os.listdir(test_dir + file)))
        test_images_list.append(images_list)

    print("------------Feature extraction for test data set------------")
    test_data = extract_features(dir=test_dir, images_list=test_images_list, files=files)
    print("------------End of extraction-------------------------------")

    print("------------Prediction on train data-------------")

    idx = [random.randint(0, len(train_data) - 1) for i in range(10)]
    test_input1 = [train_data[i] for i in idx]
    test_labels1 = [train_labels[i] for i in idx]
    test_input1 = scaler.fit_transform(test_input1)
    
    # Using the model to predict the labels of the train data
    results1 = classifier.predict(test_input1)

    # Evaluating the accuracy of the model
    
    print('predictions: ', results1)
    print("train labels: ", list(set(train_labels)))
    print("test labels: ", test_labels)
    print("Accuracy: ",(np.sum(results1 == test_labels1) / len(results1)) * 100, "%")
    #print("Accuracy: ", ((accuracy_test+accuracy_train)/2)*100,"%")
    
    ###############
    print("------------Prediction on test data-------------")

    idx = [random.randint(0, len(test_data) - 1) for i in range(10)]
    train_input2 = [test_data[i] for i in idx]
    train_labels2 = [test_labels[i] for i in idx]
    train_input2 = scaler.fit_transform(train_input2)
    
    results2 = classifier.predict(train_input2)
    
    
    print('predictions: ', results2)
    print("train labels: ", list(set(train_labels2)))
    print("test labels: ", test_labels)
    print("Accuracy: ",(np.sum(results2 == train_labels2) / len(results2)) * 100, "%")
    ##############
    

if __name__ == '__main__':
    main()
