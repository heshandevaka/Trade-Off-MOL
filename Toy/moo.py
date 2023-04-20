import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import random 

from utils import projection2simplex, find_min_norm_element

# multi-gradient cacl for each MOO method

# equal weighting (minimizing mean loss)
def grad_ew(grad_list, **kwargs):
    grad_ = torch.mean(grad_list, dim=0)

    return grad_

# MGDA
def grad_mgda(grad_list, **kwargs):
    lambd = find_min_norm_element(grad_list)

    grad_ = lambd @ grad_list

    return grad_

# MoCo
def grad_moco(grad_list, **kwargs):
    # get MoCo params 
    y = kwargs['MoCo']['y']
    lambd = kwargs['MoCo']['lambd']
    beta = kwargs['MoCo']['beta']
    gamma = kwargs['MoCo']['gamma']
    rho = kwargs['MoCo']['rho']

    # update y
    y = y - beta * ( y - grad_list)

    # update lambda
    lambd =  projection2simplex( lambd - gamma * ( y @ ( torch.transpose(y, 0, 1) @ lambd )  + rho * lambd ) )
    
    # compute multi-grad
    grad_ =  lambd @ y

    # update y, lambda
    kwargs['MoCo']['y'] = y
    kwargs['MoCo']['lambd'] = lambd

    return grad_

# MoDo
def grad_modo(grad_list, **kwargs):
    # get MoCo params 
    lambd = kwargs['MoDo']['lambd']
    gamma = kwargs['MoDo']['gamma']
    rho = kwargs['MoDo']['rho']

    # grad_list for MoDo contains two gradients
    grad1, grad2 = grad_list

    # update lambda
    lambd =  projection2simplex( lambd - gamma * ( grad1 @ ( torch.transpose(grad2, 0, 1) @ lambd )  + rho * lambd ) )
    
    # compute multi-grad
    grad_ =  0.5 * lambd @ (grad1 + grad2)

    # update lambda
    kwargs['MoDo']['lambd'] = lambd

    return grad_