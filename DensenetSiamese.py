import torch
import torch.nn as nn
import torch.nn.functional as F


class DensenetSiameseNetwork(nn.Module):
    def __init__(self):
        super(DensenetSiameseNetwork,self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.4.0', 'densenet121', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        
    def forward_once(self, x):
        
        output = self.model(x)
        return output
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        return output1, output2

class DensenetSiameseNetworkTriplet(nn.Module):
    def __init__(self):
        super(DensenetSiameseNetworkTriplet,self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.4.0', 'densenet121', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 128)
        
    def forward_once(self, x):
        
        output = self.model(x)
        return output
    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        
        return output1, output2, output3
