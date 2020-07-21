from __future__ import absolute_import
import cv2
import pandas as pd
import os
import random
import numpy as np
import keras
import multiprocessing as mp
import sklearn as sk

#Download the dataset at https://www.kaggle.com/nih-chest-xrays/data, and put the directory in the DATASET variable

DATASET = r"D:/Datasets/NIH X-Rays/data"
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
    if imagename in train_list:
        try:
            print("Training data: ", imagename)
            imagepath = findimage(imagename, DATASET)
            
            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)    
            img_array = cv2.resize(img_array, (256, 256)) #I had to do this to shrink the size down for my machine, you may not need to

            training_data = [img_array, CATEGORIES.index(label)]

        except ValueError:
            pass
            
    elif imagename in test_list:
        try:
            print("Testing data: ", imagename)
            imagepath = findimage(imagename, DATASET)
            
            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)    
            img_array = cv2.resize(img_array, (256, 256))

            testing_data = [img_array, CATEGORIES.index(label)]

        except ValueError:
            pass
    
    return training_data, testing_data

def main():
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(preprocess, data)
    
    print("Combining outputs...")
    
    training_data_total = []
    testing_data_total = []
    
    for result in results:
        training_data_total.append(result[0])
        testing_data_total.append(result[1])

    random.shuffle(training_data_total)
    random.shuffle(testing_data_total)
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for features, label in training_data_total:
        X_train.append(features)
        y_train.append(label)

    for features, label in testing_data_total:
        X_test.append(features)
        y_test.append(label)   

    X_train = np.array(X_train).reshape(-1, 256, 256, 1) #If your machine can handle the full resolution images, change this to have 1024 instead of 1024
    X_test = np.array(X_test).reshape(-1, 256, 256, 1)

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    sk.preprocessing.StandardScaler().fit_transform(X_train)
    sk.preprocessing.StandardScaler().fit_transform(X_test)

    pca = sk.decomposition.PCA()
    pca.fit_transform(X_train)
    pca.fit_transform(X_test)

    print("Saving Arrays")
    np.save("data/arrays/X_train_256_pca.npy", X_train)
    np.save("data/arrays/y_train_256_pca.npy", y_train)
    np.save("data/arrays/X_test_256_pca.npy", X_test)
    np.save("data/arrays/y_test_256_pca.npy", y_test)

if __name__ == "__main__":
    main()