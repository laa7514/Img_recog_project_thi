import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):

    def __init__(self, input_dim=45*45*3, num_classes=43):
        super(FullyConnectedNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 2048)
        self.layer2 = nn.Linear(2048, 1024)
        self.layer3 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = torch.flatten(x,1)
        x = F.relu(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x


class SmallConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, #Farbbild also RGB -> 3 Channel
                               out_channels= 12, 
                               kernel_size=(3,3),
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros',
                               device=None,
                               dtype=None)       
        # Conv1 hat eine Output Dimension von (44-3+1,44-3+1,16) = (42, 42, 16)
        self.pool1 = torch.nn.MaxPool2d(2,
                        stride=None,
                        padding=0, 
                        dilation=1,
                        return_indices=False,
                        ceil_mode=False)
        # pool1 hat eine Output Dimension von ((42+2*0-1*(2-1) +1),(42+2*0-1*(2-1) +1),16) = (42, 42, 16)
        self.conv2 = nn.Conv2d(in_channels=12, #Farbbild also RGB -> 3 Channel
                               out_channels= 15, 
                               kernel_size=(3,3),
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros',
                               device=None,
                               dtype=None)  
        self.pool2 = torch.nn.MaxPool2d(2,
                        stride=None,
                        padding=0, 
                        dilation=1,
                        return_indices=False,
                        ceil_mode=False)
        self.fc1 = nn.Linear(1215, 120)
        self.fc2 = nn.Linear(120, 43)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1) # flatten alle dimensions außer batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.logSoftmax(x)
        return x