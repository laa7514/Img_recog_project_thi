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


#### Hyperparameters ####
batch_size = 3
num_epochs = 5
lr = 0.0005
######################

data_path = '/Users/anouknormann/Desktop/PracticalDeepLearning/ImageProject/Objects_DataSet'
train_path = '/Users/anouknormann/Desktop/PracticalDeepLearning/ImageProject/Objects_DataSet/train_data'
val_path = '/Users/anouknormann/Desktop/PracticalDeepLearning/ImageProject/Objects_DataSet/val_data'






weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
Mobile_transform = weights.transforms()
#Mobile_transform = Mobile_transform()
model = models.mobilenet_v2(weights=weights)
model.eval()

#setzt alle Gewichte auf False um die Anpassung der pretrained Weights zu verhindern -> Backprop passt nur die letzte neue Schicht an
for param in model.parameters():
    param.requires_grad = False

# num_ftrs = model.classifier.in_features
num_classes = 3
model.classifier = torch.nn.Linear(1280, num_classes)



train_set = ClassroomObjectsDataset(f'{data_path}/train_data.csv', train_path, transforms=Mobile_transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_set = ClassroomObjectsDataset(f'{data_path}/val_data.csv', val_path, transforms=Mobile_transform)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)




trainer = ModelTrainer(model, train_dataloader, loss_criterion=nn.CrossEntropyLoss(), learning_rate=lr, num_epochs=num_epochs, num_classes=num_classes)
print('Training on:  ', trainer.device)
trainer.train(val_dataloader=test_dataloader)
print("")
print("Training finished. Starting evaluation.")
trainer.multiclass_test(test_dataloader)

