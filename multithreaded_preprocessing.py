import multiprocessing as mp
import os
import random
import time

import cupy as cp
import cv2
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
    fourier(image)
        Returns a Fourier transform of the image
    edgedetect(image, radius)
        Runs an edge detection algorithm on image, with radius being
        the pixel radius of the high-pass filter used for the detection
    """

    def __init__(self, dataset_dir: str, image_size: int = 256, **kwargs):
        """
        Constructs the image_size and dataset_dir attributes and loads data
        """
        self.image_size = image_size
        self.dataset_dir = dataset_dir
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
        testpath = os.path.join(self.dataset_dir, "test_list.txt")
        csvpath = os.path.join(self.dataset_dir, 'Data_Entry_2017.csv')
        trainpath = os.path.join(self.dataset_dir, "train_val_list.txt")

        csv_data = pd.read_csv(csvpath)

        with open(trainpath, "r") as train_list_file:
            train_list = train_list_file.read()

        with open(testpath, "r") as test_list_file:
            test_list = test_list_file.read()

        return csv_data, train_list, test_list

    def _findimage(self, name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

    def fourier(self, image: cp.ndarray) -> cp.ndarray:
        """Does a Discrete Fourier Transform on the input image

        Args:
            image (cp.ndarray): Input image

        Returns:
            cp.ndarray: Fourier Transformed Image
        """
        dft = cv2.dft(cp.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = cp.fft.fftshift(dft)
        return dft_shift

    def edgedetect(self, image: cp.ndarray, radius: int = 50) -> cp.ndarray:
        """Runs Fourier edge detection on image

        Args:
            image (cp.ndarray): Input image
            radius (int, optional): Radius of mask. Defaults to 50.

        Returns:
            cp.ndarray: Edges of image
        """
        rows, cols, _ = image.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = cp.ones((rows, cols, 2), cp.uint8)

        center = [crow, ccol]

        x, y = cp.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        mask[mask_area] = 0

        dft_shift = self.fourier(image)
        fshift = dft_shift * mask

        f_ishift = cp.fft.ifftshift(fshift)
        reversed = cv2.idft(f_ishift)
        output = cv2.magnitude(reversed[:, :, 0], reversed[:, :, 1])
        return output

    def _preprocess(self, row: list) -> tuple:
        """Preprocesses a single row of data

        Args:
            row (list): Row of Pandas dataframe

        Returns:
            training_data: If data is in train_list, contains cp.ndarray
                with image data and list of labels
            testing_data: If data is in test_list, contains cp.ndarray
                with image data and list of labels
        """
        row = row[1]
        imagename = row['Image Index']
        label = row['Finding Labels']

        training_data = []
        testing_data = []

        imagesize_tuple = (self.image_size, self.image_size)

        if imagename in self.train_list:
            print("Training data: ", imagename)
            imagepath = self._findimage(imagename, self.dataset_dir)

            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, imagesize_tuple)

            multilabels = label.split("|")
            labels = [label for label in multilabels]

            training_data = [img_array, labels]

        elif imagename in self.test_list:
            print("Testing data: ", imagename)
            imagepath = self._findimage(imagename, self.dataset_dir)

            img_array = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, imagesize_tuple)

            multilabels = label.split("|")
            labels = [label for label in multilabels]

            testing_data = [img_array, labels]

        return training_data, testing_data

    def start(self):
        """
        Starts the multithreaded preprocessing and saves outputs to
        data/arrays/ as X_train_{image_size}.npy, y_train_{image_size}.npy,
        X_test_{image_size}.npy, and y_test_{image_size}.npy
        """
        starttime = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(self._preprocess, self.csv_data.iterrows())
        training_data, testing_data = zip(*results)

        print("Combining outputs...")

        random.shuffle(training_data)
        random.shuffle(testing_data)

        X_train = [item[0] for item in training_data]
        y_train = [item[1] for item in training_data]
        X_test = [item[0] for item in testing_data]
        y_test = [item[1] for item in testing_data]

        resizetuple = (-1, self.image_size, self.image_size, 1)
        X_train = cp.array(X_train).reshape(resizetuple)
        X_test = cp.array(X_test).reshape(resizetuple)

        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(y_train)
        y_test = mlb.fit_transform(y_test)

        elapsed_time = time.time() - starttime
        print(f"Time taken: {elapsed_time // 60}m {elapsed_time % 60}s")

        print("Saving Arrays")
        cp.save(f"data/arrays/X_train_{self.image_size}.npy", X_train)
        cp.save(f"data/arrays/y_train_{self.image_size}.npy", y_train)
        cp.save(f"data/arrays/X_test_{self.image_size}.npy", X_test)
        cp.save(f"data/arrays/y_test_{self.image_size}.npy", y_test)

        return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    preprocessor = PreprocessImages()
    preprocessor()
