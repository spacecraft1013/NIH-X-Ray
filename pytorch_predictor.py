import torch
import argparse
import cv2
import numpy as np

CATEGORIES = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion",
              "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
              "Consolidation", "Edema", "Emphysema", "Fibrosis",
              "Pleural_Thickening", "Hernia"]

parser = argparse.ArgumentParser(description='Load models and predict things.')
parser.add_argument('image', type=str,
                    help='An image for the model to predict.')
parser.add_argument('--model', '-m', type=str,
                    help='Path to pth model to use to predict image.')
parser.add_argument('--grayscale', '-g',
                    help='Whether to read the image as grayscale, \
                          default is image default.', action='store_true')
args = parser.parse_args()

model_file = args.model

print('Loading model...', end='', flush=True)
model = torch.load(model_file)
model.eval()
print('Done!')

print('Loading image...', end='', flush=True)
shape = (256, 256)

if args.grayscale:
    img_array = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    channels = 1

else:
    img_array = cv2.imread(args.image)
    channels = 3

img_array = cv2.resize(img_array, (shape[0], shape[1]))
img_array = np.array(img_array).reshape(-1, shape[0], shape[1], channels)
img_array = np.transpose(img_array, (0, 3, 1, 2))
tensor = torch.Tensor(img_array)
print('Done!')

print("Predicting...")
prediction = model(tensor)
print(prediction)
