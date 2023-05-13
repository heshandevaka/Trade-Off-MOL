import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# testing data_loaders
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

# test onehot encoding
def _test_one_hot_enc(dataloader):
    # onehot encoding for classes
    onehot_enc = torch.eye(10)

    # sample data and dummy predict
    data, label = next(iter(dataloader))

    # print results
    print('Label:', label[:2])
    print('One-hot encoded label:', onehot_enc[label[:2]]) 

# test loss functions
def _test_loss_functions(dataloader):
    # init model
    model = Model()

    # onehot encoding for classes
    onehot_enc = torch.eye(10)

    # sample data and dummy predict
    data, label = next(iter(dataloader))
    pred = model(data)

    # some useful activations
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)

    # testing loss functions

    # cross-netropy loss (same as nll loss)
    cross_entropy_loss = nn.CrossEntropyLoss()
    print('Cross-entropy loss loss:', cross_entropy_loss(pred, label))

    # l1loss
    l1_loss = nn.L1Loss()
    print('L1 loss loss:', l1_loss(softmax(pred), onehot_enc[label]))