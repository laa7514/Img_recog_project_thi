
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import time
import pandas
from PIL import Image
import numpy as np

from model_trainer import ModelTrainer
from datasets import TrafficSignDataset

### Hyperparameters ###
batch_size = 64
num_epochs = 10
lr = 0.0001
######################

data_root_path = Path("/Users/anouknormann/Desktop/PracticalDeepLearning/Traffic Sign/traffic_sign_dataset")
train_path = data_root_path / "Train"
test_path = data_root_path / "Test"


transforms = transforms.Compose([
        transforms.Resize((44,44)),
        transforms.ToTensor()
])
weights = models.ResNet50_Weights.DEFAULT
resnet_transform = models.ResNet50_Weights.DEFAULT.transforms
resnet_transform = resnet_transform()
model = models.resnet50(weights=weights)
model.eval()

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
num_classes = 43
model.fc = torch.nn.Linear(num_ftrs, num_classes)



train_set = TrafficSignDataset(data_root_path / "Train.csv", data_root_path, transforms=resnet_transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_set = TrafficSignDataset(data_root_path / "Test.csv", data_root_path, transforms=resnet_transform)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


trainer = ModelTrainer(model, train_dataloader, loss_criterion=nn.CrossEntropyLoss(), learning_rate=lr, num_epochs=num_epochs, num_classes=43)
print('Training on:  ', trainer.device)
trainer.train(val_dataloader=test_dataloader)
print("")
print("Training finished. Starting evaluation.")
trainer.multiclass_test(test_dataloader)
