import cv2
import pandas as pd
import os
import random
import numpy as np
import multiprocessing as mp
from sklearn.preprocessing import MultiLabelBinarizer
import time

CATEGORIES = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

class PreprocessImages():
    """
    A class for preprocessing images from the NIH ChestX-ray14 Dataset. Dataset can be downloaded at https://www.kaggle.com/nih-chest-xrays/data.

    Attributes
    ----------
    image_size : int
        The one-sided dimension to resize the images to
    dataset_dir : str
        The directory of the ChestX-ray14 dataset
    csv_data : pd.Dataframe
        Contents of the Data_Entry_2017.csv file in the dataset
    train_list : str
        Contents of the train_val_list.txt file in the dataset
    test_list : str
        Contents of the test_list.txt file in the dataset

    Methods
    -------
    start()
        Starts the multithreaded preprocessing of the dataset
    load_initial_data()
        Loads the initial data needed for the preprocessing
    preprocess(row)
        Preprocesses a single row of the csv_data dataframe
    """

    def __init__(self, dataset_dir, image_size=256):
        """
        Constructs the image_size and dataset_dir attributes and runs load_initial_data()
        """
        self.image_size = image_size
        self.dataset_dir = dataset_dir
        self.csv_data, self.train_list, self.test_list = self.load_initial_data()

    def load_initial_data(self):
        """
        Loads the initial data needed for preprocessing

        Returns
        -------
        csv_data : pd.DataFrame
            Data contained in Data_Entry_2017.csv
        train_list : str
            Data contained in train_val_list.txt
        test_list : str
            Data contained in test_list.txt
        """
        csv_data = pd.read_csv(os.path.join(self.dataset_dir, 'Data_Entry_2017.csv'))

        with open(os.path.join(self.dataset_dir, "train_val_list.txt"), "r") as train_list_file:
            train_list = train_list_file.read()
        with open(os.path.join(self.dataset_dir, "test_list.txt"), "r") as test_list_file:
            test_list = test_list_file.read()
        return csv_data, train_list, test_list

    def findimage(self, name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

    def preprocess(self, row):
        """
        Preprocesses a single row of data
        
        Parameters
        ----------
        row : list
            Row of csv data from dataset
            
        Returns
        -------
        training_data : list
            If data is in train_list, contains np.ndarray with image data and list of labels
        testing_data : list
            If data is in test_list, contains np.ndarray with image data and list of labels
        """
        row = row[1]
        imagename = row['Image Index']
        label = row['Finding Labels']

        training_data = []
        testing_data = []

        if imagename in self.train_list:
            print("Training data: ", imagename)
            imagepath = self.findimage(imagename, self.datasetdir)
            
            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (self.image_size, self.image_size))
            
            multilabels = label.split("|")
            labels = [label for label in multilabels]

            training_data = [img_array, labels]

        elif imagename in self.test_list:
            print("Testing data: ", imagename)
            imagepath = self.findimage(imagename, self.datasetdir)
            
            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (self.image_size, self.image_size))

            multilabels = label.split("|")
            labels = [label for label in multilabels]

            testing_data = [img_array, labels]

        return training_data, testing_data

    def start(self):
        """
        Starts the multithreaded preprocessing and saves outputs to data/arrays/ as X_train_{image_size}.npy, y_train_{image_size}.npy, X_test_{image_size}.npy, and y_test_{image_size}
        """
        starttime = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(self.preprocess, list(self.csv_data.iterrows()))

        print("Combining outputs...")

        training_data_total = [result[0] for result in results if result[0] != []]
        testing_data_total = [result[1] for result in results if result[1] != []]

        random.shuffle(training_data_total)
        random.shuffle(testing_data_total)

        X_train = [item[0] for item in training_data_total]
        y_train = [item[1] for item in training_data_total]
        X_test = [item[0] for item in testing_data_total]
        y_test = [item[1] for item in testing_data_total]

        X_train = np.array(X_train).reshape(-1, self.image_size, self.image_size, 1)
        X_test = np.array(X_test).reshape(-1, self.image_size, self.image_size, 1)

        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(y_train)
        y_test = mlb.fit_transform(y_test)

        elapsed_time = time.time() - starttime
        print(f"Time taken: {elapsed_time // 60}m {elapsed_time % 60}s")

        print("Saving Arrays")
        np.save(f"data/arrays/X_train_{self.image_size}.npy", X_train)
        np.save(f"data/arrays/y_train_{self.image_size}.npy", y_train)
        np.save(f"data/arrays/X_test_{self.image_size}.npy", X_test)
        np.save(f"data/arrays/y_test_{self.image_size}.npy", y_test)

if __name__ == "__main__":
    preprocessor = PreprocessImages()
    preprocessor.start()