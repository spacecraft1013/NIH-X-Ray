import datetime
import multiprocessing as mp
import os
import random
from functools import partial

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

CATEGORIES = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion",
              "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
              "Consolidation", "Edema", "Emphysema", "Fibrosis",
              "Pleural_Thickening", "Hernia"]


class PreprocessImages:
    """
    A class for preprocessing images from the NIH ChestX-ray14 Dataset.
    Dataset can be downloaded at https://www.kaggle.com/nih-chest-xrays/data.

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
    """

    def __init__(self, dataset_dir: str, image_size: int = 256):
        """
        Constructs the image_size and dataset_dir attributes and loads data
        """
        # Initialize variables
        self.image_size = image_size
        self.dataset_dir = dataset_dir
        self.resizetuple = (-1, image_size, image_size, 1)

        # Call function to load metadata and training and testing image list
        self.csv_data, self.train_list, self.test_list \
            = self._load_initial_data()

    def __call__(self):
        """Calls self.start()"""
        return self.start()

    def _load_initial_data(self):
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

        # Get paths for files
        testpath = os.path.join(self.dataset_dir, "test_list.txt")
        csvpath = os.path.join(self.dataset_dir, 'Data_Entry_2017.csv')
        trainpath = os.path.join(self.dataset_dir, "train_val_list.txt")

        # Load files
        csv_data = pd.read_csv(csvpath)

        with open(trainpath, "r") as train_list_file:
            train_list = train_list_file.read()

        with open(testpath, "r") as test_list_file:
            test_list = test_list_file.read()

        return csv_data, train_list, test_list

    def _findimage(self, name, path):
        # Finds image paths from image names
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

    # Function to preprocess row for data for binary classifier with single class
    def _preprocess_single(self, classname: str, row: pd.Series) -> tuple:
        """Preprocesses a single row of data

        Args:
            row (pd.Series): Row of Pandas dataframe

        Returns:
            training_data: If data is in train_list, contains np.ndarray
                with image data and list of labels
            testing_data: If data is in test_list, contains np.ndarray
                with image data and list of labels
        """

        # Assign values of image name and label to variables
        row = row[1]
        imagename = row['Image Index']
        label = row['Finding Labels']

        # Initialize lists for training and testing data
        training_data = []
        testing_data = []

        # Create resizing tuple
        imagesize_tuple = (self.image_size, self.image_size)

        if imagename in self.train_list:
            print("Training data: ", imagename)

            # Find the image path from the image name
            imagepath = self._findimage(imagename, self.dataset_dir)

            # Load image
            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

            # Resize image
            img_array = cv2.resize(img_array, imagesize_tuple)

            # Check label for image and assign 0 or 1
            if label == 'No Finding':
                label_binary = [0]
            elif classname in label:
                label_binary = [1]

            # Add image and label to training data
            training_data = [img_array, label_binary]

        elif imagename in self.test_list:
            print("Testing data: ", imagename)

            # Find the image path from the image name
            imagepath = self._findimage(imagename, self.dataset_dir)

            # Load image
            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

            # Resize image
            img_array = cv2.resize(img_array, imagesize_tuple)

            # Check label for image and assign 0 or 1
            if label == 'No Finding':
                label_binary = [0]
            elif classname in label:
                label_binary = [1]

            # Add image and label to testing data
            testing_data = [img_array, label_binary]

        return training_data, testing_data

    def _preprocess(self, row: pd.Series) -> tuple:
        """Preprocesses a single row of data

        Args:
            row (pd.Series): Row of Pandas dataframe

        Returns:
            training_data: If data is in train_list, contains np.ndarray
                with image data and list of labels
            testing_data: If data is in test_list, contains np.ndarray
                with image data and list of labels
        """

        # Assign values of image name and label to variables
        row = row[1]
        imagename = row['Image Index']
        label = row['Finding Labels']

        # Initialize lists for training and testing data
        training_data = []
        testing_data = []

        # Create resizing tuple
        imagesize_tuple = (self.image_size, self.image_size)

        if imagename in self.train_list:
            print("Training data: ", imagename)

            # Find the image path from the image name
            imagepath = self._findimage(imagename, self.dataset_dir)

            # Load image
            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

            # Resize image
            img_array = cv2.resize(img_array, imagesize_tuple)

            # Split the labels into a list of labels
            multilabels = label.split("|")

            # Add image and label to training_data
            training_data = [img_array, multilabels]

        elif imagename in self.test_list:
            print("Testing data: ", imagename)

            # Find the image path from the image name
            imagepath = self._findimage(imagename, self.dataset_dir)

            # Load image
            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

            # Resize image
            img_array = cv2.resize(img_array, imagesize_tuple)

            # Split the labels into a list of labels
            multilabels = label.split("|")

            # Add image and label to testing data
            testing_data = [img_array, multilabels]

        return training_data, testing_data

    def start_single(self, classname: str):

        # Filter metadata csv to only contain images that either have the class or no finding
        df = self.csv_data[self.csv_data["Finding Labels"].str.contains(classname + "|No Finding")]

        starttime = datetime.datetime.now()

        # Create multiprocessing worker pool
        pool = mp.Pool(mp.cpu_count())

        # Map pool to the metadata to run the preprocessing function
        results = pool.map(partial(self._preprocess_single, classname), df.iterrows())

        # Filter out empty lists from results
        training_data = [result[0] for result in results if result[0] != []]
        testing_data = [result[1] for result in results if result[1] != []]

        print("Combining outputs...")

        # Shuffle data
        random.shuffle(training_data)
        random.shuffle(testing_data)

        # Combine images and labels into unified arrays
        X_train = [item[0] for item in training_data]
        y_train = [item[1] for item in training_data]
        X_test = [item[0] for item in testing_data]
        y_test = [item[1] for item in testing_data]

        # Reshape arrays
        X_train = np.array(X_train).reshape(self.resizetuple)
        X_test = np.array(X_test).reshape(self.resizetuple)

        endtime = datetime.datetime.now()
        print('Time taken:', endtime - starttime)

        print("Saving Arrays")

        # Save arrays to npz file
        np.savez(
            f"data/arrays/arrays_{self.image_size}_{classname.lower()}.npz",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        return (X_train, y_train), (X_test, y_test)

    def start(self):
        """
        Starts the multithreaded preprocessing and saves outputs to
        data/arrays/ as X_train_{image_size}.npy, y_train_{image_size}.npy,
        X_test_{image_size}.npy, and y_test_{image_size}.npy
        """
        starttime = datetime.datetime.now()

        # Create multiprocessing worker pool
        with mp.Pool(mp.cpu_count()) as pool:
            # Map pool to the metadata to run the preprocessing function
            results = pool.map(self._preprocess, self.csv_data.iterrows())

        # Filter out empty lists from results
        training_data = [result[0] for result in results if result[0] != []]
        testing_data = [result[1] for result in results if result[1] != []]

        print("Combining outputs...")

        # Shuffle data
        random.shuffle(training_data)
        random.shuffle(testing_data)

        # Combine images and labels into unified arrays
        X_train = [item[0] for item in training_data]
        y_train = [item[1] for item in training_data]
        X_test = [item[0] for item in testing_data]
        y_test = [item[1] for item in testing_data]

        # Reshape arrays
        X_train = np.array(X_train).reshape(self.resizetuple)
        X_test = np.array(X_test).reshape(self.resizetuple)

        # Initialize multilabelbinarizer
        mlb = MultiLabelBinarizer()

        # Use multilabelbinarizer to convert labels
        y_train = mlb.fit_transform(y_train)
        y_test = mlb.fit_transform(y_test)

        # Remove first item from labels to make no finding all zeros
        y_train = np.array([i[1:] for i in y_train])
        y_test = np.array([i[1:] for i in y_test])

        endtime = datetime.datetime.now()
        print('Time taken:', endtime - starttime)

        print("Saving Arrays")

        # Save arrays to npz file
        np.savez(
            f"data/arrays/arrays_{self.image_size}.npz",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    preprocessor = PreprocessImages("F:/Datasets/NIH X-Rays/data", 256)
    preprocessor()
