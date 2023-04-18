import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

# Define the transforms for data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# You can similarly create a test dataset if you have test data
testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


# TODO: define dense layer model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        input_dim = 72 # ?
        hidden_dim = 512
        self.resnet_network = resnet18(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                    nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        
    def forward(self, inputs):
        out = self.resnet_network(inputs)
        out = torch.flatten(self.avgpool(out), 1)
        out = self.hidden_layer(out)
        return out