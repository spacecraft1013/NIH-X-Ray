import keras
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

CATEGORIES = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

parser = argparse.ArgumentParser(description='Load models and predict things.')
parser.add_argument('image', type=str, help='An image for the model to predict.')
parser.add_argument('--model', '-m', type=str, help='Path to h5 model to use to predict image.')
parser.add_argument('--grayscale', '-g', type=bool, help='Whether to read the image as grayscale, default is image default.')
args = parser.parse_args()

model_file = args.model

model = keras.models.load_model(model_file)

shape = model.input_shape

if args.grayscale == True:
    img_array = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    channels = 1

else:
    img_array = cv2.imread(args.image)
    channels = 3

img_array = cv2.resize(img_array, (shape[1], shape[1]))
img_array = np.array(img_array).reshape(-1, shape[1], shape[1], channels)
activation_map_model = keras.models.Model(inputs=model.get_layer(1), outputs=model.get_layer(1))
activation_map_model.set_weights(model.get_weights())
activation_map = activation_map_model.predict(img_array)
overlay = cv2.addWeighted(img_array, 1, activation_map, 0.3, 0.1)

print("Predicting...")

prediction = model.predict(img_array)
print(CATEGORIES.index(np.argmax(prediction, axis=1)))
cv2.imshow(cv2.resize(overlay, (1024, 1024)))
cv2.waitKey(0)