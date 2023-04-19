import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# hyper-parameters
batch_size = 64

# Define the transforms for data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.,), (0.5,))]
)

# Load the MNIST dataset
train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

# Create test dataset
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)


# testing functions
def _test_data_loaders(dataloader):
    # get number of data
    print('Total number of data:', dataloader.batch_size * len(dataloader))
    # get label and image batch from dataset
    img_batch, label_batch = next(iter(dataloader))
    # print('Image data batch:', img) # output too bulky
    print('Image data batch dim:', img_batch.shape)
    # print('label batcha:', label) # output too bulky
    print('label batch dim:', label_batch.shape)

    i = np.random.randint(dataloader.batch_size) # 0 <= i < batchsize
    # plot sample image from batch to verify data
    plt.imshow(img_batch[i][0], cmap='Greys')
    plt.title(f'{label_batch[i]}')
    plt.savefig('test_mnist_image.png')

    # reshape test
    test_tensor = img_batch[:min(3, dataloader.batch_size)]
    # print('Tensor before reshaping:', test_tensor) # output too bulky
    reshaped_tensor = test_tensor.view(test_tensor.shape[0], -1) 
    # print('Tensor after reshaping (flatten inner dimensions):', reshaped_tensor) # output too bulky

    # test data is preserved in the flattened tensor
    plt.imshow(reshaped_tensor[2].view(28, 28), cmap='Greys')
    plt.title(f'test data reshape')
    plt.savefig('test_data_reshape.png')   

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_dim = 28 * 28 # input image data in vector form
        self.hidden_dim = 512 # hidden layer size
        self.output_dim = 10 # number of digit classes

        # define model layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, inputs):
        x = inputs.view(inputs.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

_test_data_loaders(train_dataloader)

# init model
model = Model()

# onehot encoding for classes
onehot_enc = torch.eye(10)

# sample data and dummy predict
data, label = next(iter(train_dataloader))
pred = model(data)

logsoftmax = nn.LogSoftmax(dim=1)

# cross-netropy loss (same as nll loss)
cross_entropy_loss = nn.CrossEntropyLoss()
print('Cross-entropy loss loss:', cross_entropy_loss(logsoftmax(pred), onehot_enc[label]))



print(onehot_enc[label])



