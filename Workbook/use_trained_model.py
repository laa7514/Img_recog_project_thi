import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import time
import pandas
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

from model_trainer import ModelTrainer
from datasets import ClassroomObjectsDataset
from models import Classify_image


##############################
model_path = '/Users/anouknormann/Desktop/PracticalDeepLearning/Img_recog_project_thi/model'
num_classes = 3
##############################




# Initialize a new model instance with the same architecture
new_model = models.mobilenet_v2(weights=None)  # Initialize model with the same architecture
new_model.classifier = torch.nn.Linear(1280, num_classes)  # Modify the classifier to match the number of classes

# Load the saved weights into the new model
new_model.load_state_dict(torch.load(model_path))
new_model.eval()
print('Model weights loaded into new model instance')


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])




# Classify a new image
image_path = '/Users/anouknormann/Desktop/PracticalDeepLearning/ImageProject/Objects_DataSet/train_data/img_2_2_.jpeg'
predicted_class = Classify_image(image_path, new_model, transform)
print(f'The predicted class for the image is: {predicted_class}')

