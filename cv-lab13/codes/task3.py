import random
from sklearn.preprocessing import StandardScaler
from concurrent import futures
import cv2
from functools import partial
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from skimage import feature
import numpy as np
import os
from skimage import exposure
from skimage import feature
classifier_type = "knn"  # "linear_svm", "rbf_svm", "knn", "random_forest
feature_type = "hog_and_lbp"  # "raw_pixels", "hog", "lbp", "hog_and_lbp"


# Complete this function!
def processing(feature_type, dir, images_list, idx, file):
    print("Processing: " + file)
    temp = []

    for addr in images_list[idx]:
        I = cv2.imread(os.path.join(dir, file, addr) , cv2.IMREAD_GRAYSCALE)

        if feature_type == "raw_pixels":
            temp.append(I.ravel())

        elif feature_type == "hog":
            (H, hogImage) = feature.hog(I, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
            
            hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
            hogImage = hogImage.astype("uint8")
            temp.append(hogImage.ravel())

        elif feature_type == "lbp":
            lbp = feature.local_binary_pattern(I, 28, 3)
            J = np.copy(lbp)
            J = np.array(J, dtype=np.float32)
            temp.append(J.ravel())
        elif feature_type == "hog_and_lbp":
            T = I.copy()
            (H, hogImage) = feature.hog(I, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
            
            hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
            hogImage = hogImage.astype("uint8")

            lbp = feature.local_binary_pattern(T, 28, 3)
            J = np.copy(lbp)
            J = np.array(J, dtype=np.float32)
            temp.append(hogImage.ravel()+ J.ravel())

    return temp


# Don't bother yourself with this function.
def extract_features(feature_type, dir, images_list, files):
    data = []
    with futures.ProcessPoolExecutor() as executor:
        indices = np.arange(len(images_list))
        func = partial(processing, feature_type, dir, images_list)
        results = executor.map(func, indices, files)

        for result in results:
            data.extend(result)
    return data


# Complete this function!
def train(classifier_type, train_data, train_labels):
    if classifier_type == "linear_svm":
        clf = SVC( C = 0.1 , kernel = "linear" )
    elif classifier_type == "rbf_svm":
        clf  = SVC( C = 1e+3 , gamma = 1e-4 , kernel = "rbf" )
    elif classifier_type == "knn":
        clf = KNeighborsClassifier( n_neighbors = 5 )
    elif classifier_type == "random_forest":
        clf = RandomForestClassifier( n_estimators = 200 )

    clf.fit(train_data, train_labels)
    return clf


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
    train_data = extract_features(feature_type=feature_type, dir=train_dir, images_list=train_images_list, files=files)
    print("------------End of extraction--------------------------------")

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)

    classifier = train(classifier_type, train_data, train_labels)

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
    test_data = extract_features(feature_type=feature_type, dir=test_dir, images_list=test_images_list, files=files)
    print("------------End of extraction-------------------------------")

    print("------------Prediction on test data-------------")

    test_input = scaler.fit_transform(test_data)
    results = classifier.predict(test_input)
    print('predictions: ', results)
    print("train labels: ", list(set(train_labels)))
    print("test labels: ", test_labels)
    print("Accuracy: ", (np.sum(results == test_labels) / len(results)) * 100, "%")

if __name__ == '__main__':
    main()
