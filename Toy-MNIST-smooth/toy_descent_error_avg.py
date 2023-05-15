import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import matplotlib.pyplot as plt
import numpy as np
import random 

from utils import projection2simplex, find_min_norm_element, set_seed
from moo_lambd import grad_ew, grad_mgda, grad_moco, grad_modo

# Create arg parser

parser = argparse.ArgumentParser(description='Arguments for Toy MOO task')

# general
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for the model')
parser.add_argument('--moo_method', type=str, default='EW', help='MOO method for updating model, option: EW, MGDA, MoCo, MoDo')

# moo method specific
# EW
# None
# MGDA
# None
# MoCo
parser.add_argument('--beta_moco', type=float, default=0.1, help='learning rate of tracking variable')
parser.add_argument('--gamma_moco', type=float, default=0.1, help='learning rate of lambda')
parser.add_argument('--rho_moco', type=float, default=0.0, help='regularization parameter of lambda subproblem')
# MoDo
parser.add_argument('--gamma_modo', type=float, default=0.1, help='learning rate of lambda')
parser.add_argument('--rho_modo', type=float, default=0.0, help='regularization parameter')

# parse args
params = parser.parse_args()
print(params)

# General hyper-parameters

# seed
seed = params.seed
# batch size for sampling data
batch_size = params.batch_size
# training iterations
num_epochs = params.num_epochs
# learning rate of model
lr = params.lr
# MOO method
moo_method = params.moo_method

# MOO method specific hyper-parameters

# EW
# None

# MoCo
moco_beta = params.beta_moco
moco_gamma = params.gamma_moco
moco_rho = params.rho_moco

# MoDo
modo_gamma = params.gamma_modo
modo_rho = params.rho_modo

# calc accuracy of predictions
def get_accuracy(pred, label):
    return torch.sum(torch.eq( torch.argmax(pred, dim=1), label )) / pred.shape[0]
    

# get performance measures of the learned model (at the end of training)
def get_performance(model, optimizer, dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc):
    grad_list = 0
    loss_list = 0
    acc = 0
    count = 0
    for data, label in iter(dataloader):
        pred = model(data)
        grad_list_, loss_list_ = get_grads(model, optimizer, pred, label, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
        grad_list += grad_list_
        loss_list += loss_list_
        acc += get_accuracy(pred.detach(), label)

        count += 1
    
    grad_list /= count
    loss_list /= count

    lambd_opt = find_min_norm_element(grad_list)
    multi_grad = lambd_opt @ grad_list
    
    return acc.item()/count, loss_list, torch.norm(multi_grad).item()

# get descent direction error
def get_descent_errror(model, optimizer, dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc, lambd):
    grad_list = 0
    loss_list = 0
    acc = 0
    count = 0
    for data, label in iter(dataloader):
        pred = model(data)
        grad_list_, loss_list_ = get_grads(model, optimizer, pred, label, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
        grad_list += grad_list_
        loss_list += loss_list_
        acc += get_accuracy(pred.detach(), label)

        count += 1
    
    grad_list /= count
    loss_list /= count

    lambd_opt = find_min_norm_element(grad_list)
    multi_grad_error = ( lambd_opt - lambd ) @ grad_list
    
    return torch.norm(multi_grad_error).item()


# get layer-wise parameter numbers
def get_layer_params(model):

    # init layer-wise param number list
    num_param_layer = []

    # print layer-wise parameter numbers
    print("\n"+"="*50)
    print('Model parameter count per layer')
    print("="*50)
    # get layerwise param numbers, with layer names
    for name, param in model.named_parameters():
        num_param_layer.append(param.data.numel())
        print(f'{name}', f'\t: {param.data.numel()}')
    print('Total number of parametrs :', sum(num_param_layer))
    print("-"*50)
    # return layerwise and total param numbers
    return sum(num_param_layer), num_param_layer

# get vectorized grad information
def get_grad_vec(model, num_param, num_param_layer):
    # initialize grad with a vecotr with size num. param.
    grad_vec = torch.zeros(num_param)
    # count params to put grad blocks in correct index of vector
    count = 0
    for param in model.parameters():
        # collect grad only if not None, else return zero vector
        if param.grad is not None:
            # calculate vecotr block start and end indices
            beg = 0 if count == 0 else sum(num_param_layer[:count])
            end = sum(num_param_layer[:(count+1)])
            # put flattened grad param into the vector block
            grad_vec[beg:end] = param.grad.data.view(-1)
        count += 1
    
    return grad_vec

# get gradient and loss values w.r.t each loss function
def get_grads(model, optimizer, pred, label, loss_dict, num_param, num_param_layer, softmax, onehot_enc):
    # init gradient list (to be collected one gradient for each loss)
    grad_list = []
    loss_list = []
    # to switch off retain_graph in loss.backward()
    num_loss = len(loss_dict) 
    # compute the loss w.r.t each loss function
    for k, loss_fn in enumerate(loss_dict):
        if loss_fn =='mse' or loss_fn =='huber':
            loss = loss_dict[loss_fn](softmax(pred), onehot_enc[label])
        else:
            loss = loss_dict[loss_fn](pred, label)
        # make gradient of model zero
        optimizer.zero_grad()
        # compute loss w.r.t current loss function
        loss.backward(retain_graph=True) if k < num_loss - 1 else loss.backward()
        # compute vectorized gradient
        grad_vec = get_grad_vec(model, num_param, num_param_layer)
        # collect the gradient for current loss
        grad_list.append(grad_vec)
        loss_list.append(loss.detach().item())
    
    return torch.stack(grad_list), np.array(loss_list)

# get loss values w.r.t each loss function (for each iteration calculation)
def get_loss(model, loss_dict, dataloader, softmax, onehot_enc):
    # init average loss list (to be collected one gradient for each loss function)
    loss_list = 0
    count = 0
    # compute the loss w.r.t each loss function
    for data, label in iter(dataloader):
        pred = model(data)
        loss_list_ = []
        for k, loss_fn in enumerate(loss_dict):
            if loss_fn =='mse' or loss_fn =='huber':
                loss_ = loss_dict[loss_fn](softmax(pred), onehot_enc[label])
            else:
                loss_ = loss_dict[loss_fn](pred, label)
            loss_list_.append(loss_.detach().item())
        loss_list = loss_list + np.array(loss_list_)
        count += 1
    return loss_list / count

# set multi-gradient in the model param grad
def set_grads(model, multi_grad, num_param_layer):
    # count params to put multi-grad blocks in correct model param grad
    count = 0
    for param in model.parameters():
        # put grad only if grad is initialized
        if param.grad is not None:
            # calculate vector block start and end indices
            beg = 0 if count == 0 else sum(num_param_layer[:count])
            end = sum(num_param_layer[:(count+1)])
            # put reshaped multi-grad block into model param grad
            param.grad.data = multi_grad[beg:end].view(param.data.size()).data.clone()
        count += 1   
    return 

# Define model architecture to solve toy problem

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
    
# Create toy datset and the data_loaders

# Define the transforms for data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.,), (0.5,))]
)

# set seed
seed_worker, g = set_seed(seed)

# Load the MNIST dataset 

# half batch size if MoDo
if moo_method == 'MoDo':
    batch_size = batch_size//2
# get the initial training dataset
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# split the data set into train and val
train_data, val_data = torch.utils.data.random_split(dataset, [50000, 10000])
# creare train loader for training
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
# create another train loader for evaluation
train_eval_dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
# create val loader
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
# get test dataset and create test dataloader
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)


# Set-up for training

# init model
model = Model()

# get layerwise parameter numbers
num_param, num_param_layer = get_layer_params(model)

# Defining loss functions 

# onehot encoding for classes
onehot_enc = torch.eye(10)

# some useful activations
logsoftmax = nn.LogSoftmax(dim=1)
softmax = nn.Softmax(dim=1)

# Loss functions

# cross-netropy loss (same as nll loss)
cross_entropy_loss = nn.CrossEntropyLoss()
# l1loss
l1_loss = nn.L1Loss()
# hinge loss
hinge_loss = torch.nn.MultiMarginLoss()
# MSE loss
mse_loss = torch.nn.MSELoss()
# Huber loss
huber_loss = torch.nn.HuberLoss(delta=0.1) # to make sure this is deifferent from mse

# dictionary of losses
loss_dict = {'cel':cross_entropy_loss, 'mse':mse_loss, 'huber':huber_loss}
# number of tasks
num_tasks = len(loss_dict)


# MOO method specific functions and parameters

# collection of multi-grad calculation methods
multi_grad_fn = {'EW': grad_ew, 'MGDA':grad_mgda, 'MoCo': grad_moco, 'MoDo':grad_modo}

# MOO menthod specific params

# EW
# None

# MGDA
# None

# MoCo
moco_kwargs = {'y':torch.zeros(num_tasks, num_param), 'lambd':torch.ones([num_tasks, ])/num_tasks, 'beta':moco_beta, 'gamma':moco_gamma, 'rho':moco_rho}

# MoDo
modo_kwargs = {'lambd':torch.ones(num_tasks)/num_tasks, 'gamma':modo_gamma, 'rho':modo_rho}

# add kwarg argumnets to one dictionary, so training is general to all methods
kwargs = {'EW':{}, 'MGDA':{}, 'MoCo':moco_kwargs, 'MoDo':modo_kwargs}

# optimiser (sgd)
optimizer = optim.SGD(model.parameters(), lr=lr)

# print log format
print("\n"+"="*50)
print(f'LOG FORMAT: Epoch: EPOCH | Descent Error: DESC ERROR')
print("="*50)
# init descent error avg calc
descent_error_sum = 0
count = 0
for i in range(num_epochs):
    # if MoDo, double sample
    if moo_method=='MoDo':
        grad_list = []
        loss_list = [0 for _ in range(num_tasks)]
        for j in range(2):
            # sample training data and label
            data, label = next(iter(train_dataloader))  
            # get model prediction (logits)  
            pred = model(data)
            # get gradients and loss values w.r.t each loss fn
            grad_list_, _ = get_grads(model, optimizer, pred, label, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
            grad_list.append(grad_list_)
    # or single batch sample for other methods
    else:
        # sample training data and label
        data, label = next(iter(train_dataloader))  
        # get model prediction (logits)  
        pred = model(data)
        # get gradients and loss values w.r.t each loss fn
        grad_list, _ = get_grads(model, optimizer, pred, label, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
    # calc multi-grad according to moo method
    lambdt, multi_grad = multi_grad_fn[moo_method](grad_list, **kwargs)
    # calc and report loss values every 100 epochs
    if i%50 == 0:
        # get descent error
        descent_error = get_descent_errror(model, optimizer, train_eval_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc, lambdt)
        count += 1
        descent_error_sum += descent_error   
        print(f"Epoch: {i: 6,} | Descent Error: {round(descent_error_sum / count, 10)}")

    # update model grad with the multi-grad
    set_grads(model, multi_grad, num_param_layer)
    # update model param
    optimizer.step()

# get final loss and error values
descent_error = get_descent_errror(model, optimizer, train_eval_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc, lambdt)
count += 1
descent_error_sum += descent_error      
print(f"Epoch: {i+1: 6,} | Descent Error: {round(descent_error_sum / count, 5)}")
    

# get final perforamnce measures from each dataset
train_acc, train_loss, train_ps = get_performance(model, optimizer, train_eval_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
val_acc, val_loss, val_ps = get_performance(model, optimizer, val_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
test_acc, test_loss, test_ps = get_performance(model, optimizer, test_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
print("\n"+"="*100)
print(f'PERF FORMAT: DATASET | Acuracy: ACC | Loss: {" ".join([f"LOSS{i+1}" for i in range(num_tasks)])} | PS: PS')
print("="*100)
print(f"Train | Acuracy: {train_acc *100 : 2.2f}% | Loss: { ' '.join(str(round(num, 5)) for num in train_loss) } | PS: {round(train_ps, 5)} ")
print(f"Val   | Acuracy: {val_acc*100 : 2.2f}% | Loss: { ' '.join(str(round(num, 5)) for num in val_loss) } | PS: {round(val_ps, 5)} ")
print(f"Test  | Acuracy: {test_acc*100 : 2.2f}% | Loss: { ' '.join(str(round(num, 5)) for num in test_loss) } | PS: {round(test_ps, 5)} ")
print("-"*100)
print(f'Optimization error  : {round(train_ps, 5)}')
print(f'Population error    : {round(test_ps, 5)}')
print(f'Generalization error: {round(test_ps  - train_ps, 5)}')

