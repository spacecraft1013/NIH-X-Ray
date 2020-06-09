from __future__ import absolute_import
import cv2
import pandas as pd
import os
import random
import numpy as np
import pickle
import keras
import multiprocessing as mp

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



def preprocess(lines):
    training_data = []
    testing_data = []

    for row in lines.iterrows():
        row = row[1]
        imagename = row['Image Index']
        label = row['Finding Labels']
        if imagename in train_list:
            try:
                print("Training data: ", imagename)
                imagepath = findimage(imagename, DATASET)
            
                img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)    
                img_array = cv2.resize(img_array, (128, 128)) #I had to do this to shrink the size down for my machine, you may not need to

                training_data.append([img_array, CATEGORIES.index(label)])

            except ValueError:
                pass
            
        elif imagename in test_list:
            try:
                print("Testing data: ", imagename)
                imagepath = findimage(imagename, DATASET)
            
                img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)    
                img_array = cv2.resize(img_array, (128, 128))

                testing_data.append([img_array, CATEGORIES.index(label)])

            except ValueError:
                pass

    random.shuffle(training_data)
    random.shuffle(testing_data)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for features, label in training_data:
        X_train.append(features)
        y_train.append(label)

    for features, label in testing_data:
        X_test.append(features)
        y_test.append(label)

    return X_train, y_train, X_test, y_test

def main():
    print("Creating Dataset")
    pool = mp.Pool(mp.cpu_count())
    extra = len(data) % mp.cpu_count() #Find the remainder after dividing the data among the threads
    segment_list = list(np.split(data[0:len(data)-extra], mp.cpu_count())) #Split the data, excluding the extra
    segment_list[0].append(data[len(data)-extra:len(data)]) #Add the extra data back to one of the thread's segment
    print(segment_list)
    results = pool.map(preprocess, segment_list)

    print("Combining outputs...")
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for result in results:
        for value in result[0]:
            X_train.append(value)
        for value in result[1]:
            y_train.append(value)
        for value in result[2]:
            X_test.append(value)
        for value in result[3]:
            y_test.append(value)

    X_train = np.array(X_train).reshape(-1, 128, 128, 1) #If your machine can handle the full resolution images, change this to have 128 instead of 128
    X_test = np.array(X_test).reshape(-1, 128, 128, 1)

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    print("Saving Arrays")
    pickle.dump(X_train, open("data/arrays/X_train_128.pickle", "wb"), protocol=4)
    pickle.dump(y_train, open("data/arrays/y_train_128.pickle", "wb"), protocol=4)
    pickle.dump(X_test, open("data/arrays/X_test_128.pickle", "wb"), protocol=4)
    pickle.dump(y_test, open("data/arrays/y_test_128.pickle", "wb"), protocol=4)

if __name__ == "__main__":
    main()