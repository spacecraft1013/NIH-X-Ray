from __future__ import absolute_import
import cv2
import pandas as pd
import os
import random
import numpy as np
import multiprocessing as mp
import sklearn as sk
from sklearn.preprocessing import MultiLabelBinarizer

#Download the dataset at https://www.kaggle.com/nih-chest-xrays/data, and put the directory in the DATASET variable

DATASET = "F:/Datasets/NIH X-Rays/data"
CATEGORIES = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

def findimage(name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

data = pd.read_csv(os.path.join(DATASET, 'Data_Entry_2017.csv'))

test_list_dir = os.path.join(DATASET, "test_list.txt")
train_list_dir = os.path.join(DATASET, "train_val_list.txt")
test_list_file = open(test_list_dir, "r")
train_list_file = open(train_list_dir, "r")
test_list = test_list_file.read()
train_list = train_list_file.read()
test_list_file.close()
train_list_file.close()

def preprocess(row):
    row = row[1]
    imagename = row['Image Index']
    label = row['Finding Labels']

    training_data = []
    testing_data = []

    if imagename in train_list:
        print("Training data: ", imagename)
        imagepath = findimage(imagename, DATASET)
        
        img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (256, 256)) #I had to do this to shrink the size down for my machine, you may not need to
        
        multilabels = label.split("|")
        labels = []
        for label in multilabels:
            labels.append(CATEGORIES.index(label))

        training_data = [img_array, labels]

    elif imagename in test_list:
        print("Testing data: ", imagename)
        imagepath = findimage(imagename, DATASET)
        
        img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (256, 256))

        multilabels = label.split("|")
        labels = []
        for label in multilabels:
            labels.append(CATEGORIES.index(label))

        testing_data = [img_array, labels]

    return training_data, testing_data

def main():
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(preprocess, list(data.iterrows()))

    print("Combining outputs...")

    training_data_total = []
    testing_data_total = []

    for result in results:
        training_data_total.append(result[0])
        testing_data_total.append(result[1])

    training_data_total = [x for x in training_data_total if x != []]
    testing_data_total = [x for x in testing_data_total if x != []]

    random.shuffle(training_data_total)
    random.shuffle(testing_data_total)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for item in training_data_total:
        X_train.append(item[0])
        y_train.append(item[1])

    for item in testing_data_total:
        X_test.append(item[0])
        y_test.append(item[1])

    X_train = np.array(X_train).reshape(-1, 256, 256, 1) #If your machine can handle the full resolution images, change this to have 1024 instead of 1024
    X_test = np.array(X_test).reshape(-1, 256, 256, 1)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.fit_transform(y_test)

    print("Saving Arrays")
    np.save("data/arrays/X_train_256.npy", X_train)
    np.save("data/arrays/y_train_256.npy", y_train)
    np.save("data/arrays/X_test_256.npy", X_test)
    np.save("data/arrays/y_test_256.npy", y_test)

if __name__ == "__main__":
    main()