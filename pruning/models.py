import torch
import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d 



class polynom_act(nn.Module):

    def __init__(self, alpha=None, beta=None, c=None):
        super(polynom_act, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        return (self.alpha * (x ** 2) + self.beta * x + self.c)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])
        

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = MaskedConv2d(1, 8, kernel_size=5, stride=1)
        nn.init.xavier_uniform(self.conv1.weight)
        
        #self.relu1 = nn.ReLU(inplace=True)
        #self.maxpool1 = nn.MaxPool2d(2)
        self.relu1 = polynom_act()
        self.avgpoo11=nn.AvgPool2d(kernel_size=2)

        self.conv2 = MaskedConv2d(8, 16, kernel_size=5, stride=1)
        nn.init.xavier_uniform(self.conv2.weight)
        
        #self.relu2 = nn.ReLU(inplace=True)
        #self.maxpool2 = nn.MaxPool2d(2)
        self.relu2= polynom_act()
        self.avgpoo11=nn.AvgPool2d(kernel_size=2)

        self.conv3 = MaskedConv2d(16, 120, kernel_size=5, stride=1)
        nn.init.xavier_uniform(self.conv3.weight)
        
        #self.relu3 = nn.ReLU(inplace=True)
        self.relu3= polynom_act()

        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84,10)
        
    def forward(self, x):
        out = self.avgpool1(self.relu1(self.conv1(x)))
        out = self.avgpool2(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = torch.flatten(out,1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        self.conv2.set_mask(torch.from_numpy(masks[1]))
        self.conv3.set_mask(torch.from_numpy(masks[2]))
        self.linear1.set_mask(torch.from_numpy(masks[3]))
        self.linear2.set_mask((torch.from_numpy(masks[3]))