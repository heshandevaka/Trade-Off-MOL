"""Adaptation of code from: https://github.com/Cranial-XIX/CAGrad"""

from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm
from tqdm import tqdm
from scipy.optimize import minimize, Bounds, minimize_scalar

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import torch
import torch.nn as nn
import seaborn as sns
import sys
import os

################################################################################
#
# Define the Optimization Problem
#
################################################################################
LOWER = 0.000005

class Toy(nn.Module):
    def __init__(self, num_data=5000, data_mean=0, data_sig=0.05, grad_mean=0, grad_sig=0.05):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([
            [-3.0, 0],
            [3.0, 0]])
        
        self.num_data = num_data # number of empirical data (sampled from population distribution)
        self.data_mean = data_mean # mean of population data distribution
        self.data_sig = data_sig # std of population distribution for sampling data
        self.grad_mean = grad_mean # mean of normal distribution for sampling noise for gradient
        self.grad_sig = grad_sig # std of normal distribution for sampling noise for gradient
        self.emp_data_set = torch.normal(self.data_mean, self.data_sig, size=(self.num_data, 2)).detach()
        self.emp_data_mean = torch.mean(self.emp_data_set, dim=0).view([2, 1])

        print(f"num_data: {num_data}, data_mean: {data_mean}, data_sig: {data_sig}, grad_mean:{grad_mean}, grad_sig:{grad_sig}")

    def forward(self, x, compute_grad=False, data_type='pop', stoch_type='grad', batch_size=1): # data types: 'stoch', 'emp', 'pop'
        x1 = x[0]
        x2 = x[1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6 
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = ( f1 * c1 + f1_sq * c2 )*2
        f2 = f2 * c1 + f2_sq * c2

        f = torch.tensor([f1, f2])
        if compute_grad:
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            g = torch.Tensor([[g11, g21], [g12, g22]])
            if data_type=='stoch':
                if stoch_type=='data':
                    batch_mean = torch.mean(self.emp_data_set[np.random.choice(np.arange(self.num_data), batch_size), :], dim=0).view([2, 1])
                elif stoch_type=='grad':
                    batch_mean = torch.mean(torch.normal(self.grad_mean, self.grad_sig, size=(batch_size, 2, 2)).detach(), dim=0).view([2, 2])
                g = g + batch_mean
            if data_type=='emp':
                g = g + self.emp_data_mean
            return f, g
        else:
            if data_type=='emp':
                f = f + torch.sum( x * self.emp_data_mean.view(-1))
            return f

    def batch_forward(self, x, data_type='pop', compute_grad=False, batch_size=2): # data types: 'emp', 'pop'

        if compute_grad:
            g = []
            g_emp = []
            f = []
            for i, x_ in enumerate(x):

                x1_, x2_ = x_[0], x_[1]

                f1_ = torch.clamp((0.5*(-x1_-7)-torch.tanh(-x2_)).abs(), LOWER).log() + 6 
                f2_ = torch.clamp((0.5*(-x1_+3)+torch.tanh(-x2_)+2).abs(), LOWER).log() + 6
                c1_ = torch.clamp(torch.tanh(x2_*0.5), 0)

                f1_sq_ = ((-x1_+7).pow(2) + 0.1*(-x2_-8).pow(2)) / 10 - 20
                f2_sq_ = ((-x1_-7).pow(2) + 0.1*(-x2_-8).pow(2)) / 10 - 20
                c2_ = torch.clamp(torch.tanh(-x2_*0.5), 0)

                f1_ = ( f1_ * c1_ + f1_sq_ * c2_ )*2
                f2_ = f2_ * c1_ + f2_sq_ * c2_

                g11_ = torch.autograd.grad(f1_, x1_, retain_graph=True)[0].item()
                g12_ = torch.autograd.grad(f1_, x2_, retain_graph=True)[0].item()
                g21_ = torch.autograd.grad(f2_, x1_, retain_graph=True)[0].item()
                g22_ = torch.autograd.grad(f2_, x2_, retain_graph=True)[0].item()
                g_ = torch.Tensor([[g11_, g21_], [g12_, g22_]])
                # if data_type=='stoch':
                #     batch_mean = torch.mean(self.emp_data_set[np.random.choice(np.arange(self.num_data), batch_size), :], dim=0).view([2, 1])
                #     g_ = g_ + batch_mean
                # if data_type=='emp':
                #     g_ = g_ + self.emp_data_mean
                g.append(g_.clone())
                g_emp.append(g_.clone()+ self.emp_data_mean)
                f.append(torch.tensor([f1_.clone(), f2_.clone()]))
            
            return torch.stack(f), torch.stack(g), torch.stack(g_emp)
        
        else:
            x1 = x[:,0]
            x2 = x[:,1]

            f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
            f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
            c1 = torch.clamp(torch.tanh(x2*0.5), 0)

            f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
            f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
            c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

            f1 = ( f1 * c1 + f1_sq * c2 )*2
            f2 = f2 * c1 + f2_sq * c2

            # for plotting emperical objective
            if data_type == 'emp':
                print('torch.sum( x * self.emp_data_mean.view(-1), dim=1)', torch.sum( x * self.emp_data_mean.view(-1), dim=1).shape)
                f1 = f1 + torch.sum( x * self.emp_data_mean.view(-1), dim=1)
                f2 = f2 + torch.sum( x * self.emp_data_mean.view(-1), dim=1)
            
            f  = torch.cat([f1.view(-1, 1), f2.view(-1,1)], -1)

            return f

class Toy_gen(nn.Module):
    def __init__(self, num_data=5000, data_mean=0, data_sig=0.05, grad_mean=0, grad_sig=0.05):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([
            [-3.0, 0],
            [3.0, 0]])
        
        self.num_data = num_data # number of empirical data (sampled from population distribution)
        self.data_mean = data_mean # mean of population data distribution
        self.data_sig = data_sig # std of population distribution for sampling data
        self.grad_mean = grad_mean # mean of normal distribution for sampling noise for gradient
        self.grad_sig = grad_sig # std of normal distribution for sampling noise for gradient
        self.emp_data_set = torch.normal(self.data_mean, self.data_sig, size=(self.num_data, 2)).detach()
        self.emp_data_mean = torch.mean(self.emp_data_set, dim=0).view([2, 1])

        print(f"num_data: {num_data}, data_mean: {data_mean}, data_sig: {data_sig}, grad_mean:{grad_mean}, grad_sig:{grad_sig}")

    def forward(self, x, compute_grad=False, data_type='pop', stoch_type='grad', batch_size=1): # data types: 'stoch', 'emp', 'pop'
        x1 = x[0]
        x2 = x[1]

        f1 = x1**2 + x2**2
        f2 = x1**2 + x2**2 + x1 + 2*x2

        f = torch.tensor([f1, f2])
        if compute_grad:
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            g = torch.Tensor([[g11, g21], [g12, g22]])
            if data_type=='stoch':
                if stoch_type=='data':
                    batch_mean = torch.mean(self.emp_dasta_set[np.random.choice(np.arange(self.num_data), batch_size), :], dim=0).view([2, 1])
                elif stoch_type=='grad':
                    batch_mean = torch.mean(torch.normal(self.grad_mean, self.grad_sig, size=(batch_size, 2, 2)).detach(), dim=0).view([2, 2])
                g = g + batch_mean
            if data_type=='emp':
                g = g + self.emp_data_mean
            return f, g
        else:
            if data_type=='emp':
                f = f + torch.sum( x * self.emp_data_mean.view(-1))
            return f

    def batch_forward(self, x, data_type='pop', compute_grad=False, batch_size=2): # data types: 'emp', 'pop'

        if compute_grad:
            g = []
            g_emp = []
            f = []
            for i, x_ in enumerate(x):

                x1_, x2_ = x_[0], x_[1]

                f1_ = torch.clamp((0.5*(-x1_-7)-torch.tanh(-x2_)).abs(), LOWER).log() + 6 
                f2_ = torch.clamp((0.5*(-x1_+3)+torch.tanh(-x2_)+2).abs(), LOWER).log() + 6
                c1_ = torch.clamp(torch.tanh(x2_*0.5), 0)

                f1_sq_ = ((-x1_+7).pow(2) + 0.1*(-x2_-8).pow(2)) / 10 - 20
                f2_sq_ = ((-x1_-7).pow(2) + 0.1*(-x2_-8).pow(2)) / 10 - 20
                c2_ = torch.clamp(torch.tanh(-x2_*0.5), 0)

                f1_ = ( f1_ * c1_ + f1_sq_ * c2_ )*2
                f2_ = f2_ * c1_ + f2_sq_ * c2_

                g11_ = torch.autograd.grad(f1_, x1_, retain_graph=True)[0].item()
                g12_ = torch.autograd.grad(f1_, x2_, retain_graph=True)[0].item()
                g21_ = torch.autograd.grad(f2_, x1_, retain_graph=True)[0].item()
                g22_ = torch.autograd.grad(f2_, x2_, retain_graph=True)[0].item()
                g_ = torch.Tensor([[g11_, g21_], [g12_, g22_]])
                # if data_type=='stoch':
                #     batch_mean = torch.mean(self.emp_data_set[np.random.choice(np.arange(self.num_data), batch_size), :], dim=0).view([2, 1])
                #     g_ = g_ + batch_mean
                # if data_type=='emp':
                #     g_ = g_ + self.emp_data_mean
                g.append(g_.clone())
                g_emp.append(g_.clone()+ self.emp_data_mean)
                f.append(torch.tensor([f1_.clone(), f2_.clone()]))
            
            return torch.stack(f), torch.stack(g), torch.stack(g_emp)
        
        else:
            x1 = x[:,0]
            x2 = x[:,1]

            f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
            f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
            c1 = torch.clamp(torch.tanh(x2*0.5), 0)

            f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
            f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
            c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

            f1 = ( f1 * c1 + f1_sq * c2 )*2
            f2 = f2 * c1 + f2_sq * c2

            # for plotting emperical objective
            if data_type == 'emp':
                print('torch.sum( x * self.emp_data_mean.view(-1), dim=1)', torch.sum( x * self.emp_data_mean.view(-1), dim=1).shape)
                f1 = f1 + torch.sum( x * self.emp_data_mean.view(-1), dim=1)
                f2 = f2 + torch.sum( x * self.emp_data_mean.view(-1), dim=1)
            
            f  = torch.cat([f1.view(-1, 1), f2.view(-1,1)], -1)

            return f

################################################################################
#
# Plot Utils
#
################################################################################

def plotme(F, all_traj=None, xl=11):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    print(torch.max(Ys))

    colormaps = {
        "sgd": "tab:blue",
        "pcgrad": "tab:orange",
        "mgd": "tab:cyan",
        "cagrad": "tab:red",
        "moco": "tab:olive",
    }

    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    c = plt.contour(X, Y, Ys[:,0].view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.title("L1(x)")

    plt.subplot(132)
    c = plt.contour(X, Y, Ys[:,1].view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.title("L2(x)")

    plt.subplot(133)
    c = plt.contour(X, Y, Ys.mean(1).view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.legend()
    plt.title("0.5*(L1(x)+L2(x))")

    plt.tight_layout()
    plt.savefig(f"toy_ct.png")

def plot3d(F, xl=11, data_type='pop'): # data types: 'emp', 'pop'
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs, data_type)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.mean(1).view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=cm.viridis)
    
    zmin = Ys.mean(1).min()
    zmax = Ys.mean(1).max()

    ax.set_zticks([-16, -8, 0, 8]) # Original
    ax.set_zlim(-20, 10) # Original
    # ax.set_zticks([zmin, zmin/2, 0, zmax/2, zmax])
    # ax.set_zlim(zmin-1, zmax+1)

    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax.view_init(25)
    plt.tight_layout()
    plt.savefig(f"./imgs/_3d-obj-{data_type}.png", dpi=1000)

# plot Pareto statioanrity at each point of the support
def plot3d_PS(F, xl=11): # data types: 'emp', 'pop'
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.tensor(np.transpose(np.array([list(X.flat), list(Y.flat)])), dtype=torch.double, requires_grad=True)

    Ys, Gs, Gs_emp = F.batch_forward(Xs, compute_grad=True)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    PS = []
    PS_emp = []
    for k, g, g_emp in enumerate(zip(Gs, Gs_emp)):
        g_mgd_ = mgd(g)
        PS.append(torch.norm(g_mgd_))
        g_mgd_ = mgd(g_emp)
        PS_emp.append(torch.norm(g_mgd_))

    PS = torch.stack(PS).view(n,n)
    PS_emp = torch.stack(PS_emp).view(n,n)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    Yv = Ys.mean(1).view(n,n)

    surf = ax.plot_surface(X, Y, PS.numpy(), cmap=cm.viridis)
    
    zmin = Ys.mean(1).min()
    zmax = Ys.mean(1).max()

    ax.set_zticks([-16, -8, 0, 8]) # Original
    ax.set_zlim(-20, 10) # Original
    # ax.set_zticks([zmin, zmin/2, 0, zmax/2, zmax])
    # ax.set_zlim(zmin-1, zmax+1)

    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax.view_init(25)
    plt.tight_layout()
    plt.savefig(f"./imgs/_3d-PS-pop.png", dpi=1000)

    # plot 2d contours
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = plt.contour(X, Y, PS, cmap=cm.viridis, linewidths=4.0, linestyles='dashed')
    c1 = plt.contour(X, Y, PS_emp, cmap=cm.viridis, linewidths=4.0)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.yticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.tight_layout()
    plt.savefig(f"./imgs/_2d-PS.png", dpi=100)
    plt.close()


def plot_contour(F, emp=False, task=1, traj=None, xl=11, plotbar=False, name="tmp"): 
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    Ys_emp = F.batch_forward(Xs, data_type='emp')

    cmap = cm.get_cmap('viridis')
    
    # Added this block to remove hardcoded calcs
    Ysmean = Ys.mean(1).numpy()
    Ysmin = np.min(Ysmean)
    Ysargmin = np.argmin(Ysmean)
    meshy = np.argmin(Ysmean)//n
    meshx = np.argmin(Ysmean)%n
    # print(meshy, Y[meshy])
    # print(Ysmin)
    # print(Ysargmin)
    # print(Ysmean.shape)
    # print(meshx, meshy, y[meshx],y[meshy])

    # yy = -8.3552 # Original
    yy = y[meshy]
    xx = x[meshx]
    # plot mean objective
    if task == 0:
        # get mean of objectives
        Yv = Ys.mean(1)
        Yv_emp = Ys_emp.mean(1)
        # intial points
        plt.plot(-8.5, 7.5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot(-8.5, -5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot( 9, 9, marker='o', markersize=10, zorder=5, color='k')
        # pareto front
        plt.plot([-7, 7], [yy, yy], linewidth=8.0, zorder=0, color='gray')
        # optimum of mean loss
        # plt.plot(0, yy, marker='*', markersize=15, zorder=5, color='k') # Original
        plt.plot(xx, yy, marker='*', markersize=15, zorder=5, color='k')
    # plot objective 1
    elif task == 1:
        # get first objective values
        Yv = Ys[:,0]
        Yv_emp = Ys_emp[:, 0]
        # optimum of loss 1
        plt.plot(7, yy, marker='*', markersize=15, zorder=5, color='k')
    # plot objective 2
    else:
        # get second objective values
        Yv = Ys[:,1]
        Yv_emp = Ys_emp[:, 1]
        # optimum of loss2
        plt.plot(-7, yy, marker='*', markersize=15, zorder=5, color='k')

    c = plt.contour(X, Y, Yv.view(n,n), cmap=cm.viridis, linewidths=4.0)
    if emp:
        c_emp = plt.contour(X, Y, Yv_emp.view(n,n), cmap=cm.viridis, linewidths=4.0)

    if traj is not None:
        for tt in traj:
            l = tt.shape[0]
            color_list = np.zeros((l,3))
            color_list[:,0] = 1.
            color_list[:,1] = np.linspace(0, 1, l)
            #color_list[:,2] = 1-np.linspace(0, 1, l)
            ax.scatter(tt[:,0], tt[:,1], color=color_list, s=6, zorder=10)

    if plotbar:
        cbar = fig.colorbar(c, ticks=[-15, -10, -5, 0, 5])
        cbar.ax.tick_params(labelsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.yticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=100)
    plt.close()

def smooth(x, n=20):
    l = len(x)
    y = []
    for i in range(l):
        ii = max(0, i-n)
        jj = min(i+n, l-1)
        v = np.array(x[ii:jj]).astype(np.float64)
        if i < 3:
            y.append(x[i])
        else:
            y.append(v.mean())
    return y

def plot_loss(trajs, name="tmp"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormaps = {
        "sgd": "tab:blue",
        "pcgrad": "tab:orange",
        "mgd": "tab:purple",
        "cagrad": "tab:red",
        "tracking": "tab:olive",
    }
    maps = {
        "sgd" : "Adam",
        "pcgrad" : "PCGrad",
        "mgd" : "MGDA",
        "cagrad" : "CAGrad",
        "tracking": "Tracking",
    }
    for method in ["sgd", "mgd", "pcgrad", "cagrad", "tracking"]:
        traj = trajs[method][::100]
        Ys = F.batch_forward(traj)
        x = np.arange(traj.shape[0])
        #y = torch.cummin(Ys.mean(1), 0)[0]
        y = Ys.mean(1) # 0.5*L1+0.5*L2 (original)
        # y = Ys[:,0] # L1
        # y = Ys[:,1] # L2

        ax.plot(x, smooth(list(y)),
                color=colormaps[method],
                linestyle='-',
                label=maps[method], linewidth=4.)

    plt.xticks([0, 200, 400, 600, 800, 1000],
               ["0", "20K", "40K", "60K", "80K", "100K"],
               fontsize=15)

    plt.yticks(fontsize=15)
    ax.grid()
    plt.legend(fontsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=100)
    plt.close()

def plot_error_norm(traj_norms, name="error_norms", cum_sam=None):

    colormaps = {
        "smgd": "tab:blue", #"#fc8d62", "#FFAC1E"
        "smgd-minibatch": "#EA5F94",#8da0cb",
        "mgd": "tab:orange",
        "cagrad": "tab:red",
        "pcgrad": "tab:green",
        "tracking": "#9D02D7" #66c2a5",
    }
    maps = {
        "smgd" : "SMG",
        "smgd-minibatch" : "SMG + increasing batch size",
        "mgd" : "MGDA",
        "cagrad" : "CAGrad",
        "pcgrad" : "PCGrad",
        "tracking": "MoCo (ours)",
    }
    linstylemaps = {
        "smgd": "--",
        "smgd-minibatch": "--",
        "mgd": "--",
        "cagrad": ":",
        "pcgrad": "-.",
        "tracking": "solid",
    }
    fig = plt.figure()
    plt.style.use('ggplot')
    ax = fig.add_subplot(111)
    for method in [ "mgd", "smgd", "pcgrad", "cagrad", "tracking"]: # ["sgd", "mgd", "pcgrad", "cagrad", "tracking"], ["smgd", "smgd-minibatch", "tracking"]

        traj_norm_method = torch.stack(traj_norms[method])

        traj_norm_std  = torch.std(traj_norm_method[:,::100], 0, unbiased=False)
        traj_norm = torch.mean(traj_norm_method[:,::100], 0)
        # Ys = F.batch_forward(traj_norm)
        # print(len(traj_norm))

        x = 100*np.arange(traj_norm.size()[0])
        #y = torch.cummin(Ys.mean(1), 0)[0]
        # y = Ys.mean(1) # 0.5*L1+0.5*L2 (original)
        # y = Ys[:,0] # L1
        # y = Ys[:,1] # L2

        ax.semilogy(x, smooth(traj_norm_method[:,::100][0]), #list(traj_norm)
                color=colormaps[method],
                linestyle=linstylemaps[method],
                label=maps[method], linewidth=4.)


        
        # ax.fill_between(x, np.array(smooth(list(traj_norm))) - np.array(smooth(list(traj_norm_std))), np.array(smooth(list(traj_norm))) + np.array(smooth(list(traj_norm_std))), color=colormaps[method],alpha=0.5)

    plt.xticks([0, 20000, 40000, 60000],
               ["0", "20K", "40K", "60K"],
               fontsize=15)
    # plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)
    # ax.grid("False")
    if '1' in name:
        plt.legend(fontsize=15)
    if '0' in name:
        plt.ylabel(r"$\Vert \nabla F(x)\lambda^*(x) \Vert$", fontsize=15) #"Multi-gradient error"
    plt.xlabel("Iterations", fontsize=15)
    #     plt.yticks([])
    #     plt.ylabels
    # plt.grid()

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{name}-iterations.png", dpi=100)
    plt.close()

    if cum_sam is not None:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # N = traj_norms["smgd"][0].shape[0]
        # sam = [i//10000+1 for i in np.arange(N)]
        # cumsam = np.array([np.sum(np.array(sam)[:i+1]) for i in np.arange(N)])[::100]
        for method in ["smgd", "smgd-minibatch", "tracking"]: # ["sgd", "mgd", "pcgrad", "cagrad", "tracking"]
            # print(traj_norms)
            
            traj_norm_method = torch.stack(traj_norms[method])

            traj_norm_std  = torch.std(traj_norm_method[:,::100], 0, unbiased=False)
            traj_norm = torch.mean(traj_norm_method[:,::100], 0)

            if method=="smgd-minibatch":            
                x = cum_sam
            else:
                x = 100*np.arange(traj_norm.size()[0])
            #y = torch.cummin(Ys.mean(1), 0)[0]
            # y = Ys.mean(1) # 0.5*L1+0.5*L2 (original)
            # y = Ys[:,0] # L1
            # y = Ys[:,1] # L2

            ax.loglog(x, smooth(list(traj_norm)),
                    color=colormaps[method],
                    linestyle=linstylemaps[method],
                    label=maps[method], linewidth=4.)

            # ax.fill_between(x, np.array(smooth(list(traj_norm))) - np.array(smooth(list(traj_norm_std))), np.array(smooth(list(traj_norm))) + np.array(smooth(list(traj_norm_std))), color=colormaps[method],alpha=0.5)


        # plt.xticks([0, 200, 400, 600, 800, 1000],
        #            ["0", "20K", "40K", "60K", "80K", "100K"],
        #            fontsize=15)
        plt.xticks(fontsize=15)
        # print(range(0, sum(sam)+1, 1000))
        plt.yticks(fontsize=15)
        # ax.grid("False")
        # if '2' in name:
        #     plt.legend(fontsize=15)
        if '0' in name:
            plt.ylabel("Multi-gradient error", fontsize=15)
        plt.xlabel("Samples", fontsize=15)
        #     plt.yticks([])
        #     plt.ylabels
        # plt.grid()

        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.tight_layout()
        plt.savefig(f"{name}-samples.png", dpi=100)
        plt.close()

# plotting utils
def plot_2d_pareto(method, out_path=""):

    t1 = torch.load(f"./{out_path}/toy0-runs1.pt")
    t2 = torch.load(f"./{out_path}/toy1-runs1.pt")
    t3 = torch.load(f"./{out_path}/toy2-runs1.pt")
    t4 = torch.load(f"./{out_path}/toy3-runs1.pt")
    t5 = torch.load(f"./{out_path}/toy4-runs1.pt")

    trajectories = {1:t1, 2:t2, 3:t3, 4:t4, 5:t5}

    fig, ax = plt.subplots(figsize=(6, 5))

    F = Toy()

    losses = []
    for res in trajectories.values():
        losses.append(F.batch_forward(res[method][0])) #losses.append(F.batch_forward(torch.from_numpy(res[method]))) # CHANGED

    yy = -8.3552
    x = np.linspace(-7, 7, 1000)

    inpt = np.stack((x, [yy] * len(x))).T
    Xs = torch.from_numpy(inpt).double()

    Ys = F.batch_forward(Xs)
    ax.plot(
        Ys.numpy()[:, 0],
        Ys.numpy()[:, 1],
        "-",
        linewidth=8,
        color="#72727A",
        label="Pareto Front",
    )  # Pareto front

    for i, tt in enumerate(losses):
        ax.scatter(
            tt[0, 0],
            tt[0, 1],
            color="k",
            s=150,
            zorder=10,
            label="Initial Point" if i == 0 else None,
        )
        colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, tt.shape[0]))
        ax.scatter(tt[:, 0], tt[:, 1], color=colors, s=5, zorder=9)

    sns.despine()
    ax.set_xlabel(r"$f_1$", size=30)
    if "gamm-0.0001" in out_path:
        ax.set_ylabel(r"$f_2$", size=30)
    # ax.xaxis.set_label_coords(1.015, -0.03)
    # ax.yaxis.set_label_coords(-0.01, 1.01)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.tight_layout()

    title_map = {
        "nashmtl": "Nash-MTL",
        "cagrad": "CAGrad",
        "mgd": "MGDA",
        "pcgrad": "PCGrad",
        "smgd": "SMG",
        "tracking": "SMGDC (ours)",
        "sgd": "Mean",
    }

    gamma_list = ["gamm-0.0001", "gamm-0.001", "gamm-0.01", "gamm-0.1", "gamm-1"]
    file_pref = ''
    for gamma in gamma_list:
        if gamma in out_path:
            file_pref = gamma

    if "gamm-0.0001" not in out_path:
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    if "gamm-1" in out_path:
        legend = ax.legend(
            loc=2, bbox_to_anchor=(-0.15, 1.3), frameon=False, fontsize=20, ncol=2
        )


        # ax.set_title(title_map[method], fontsize=25)
        plt.savefig(
            out_path + f"{file_pref}-{method}-os.pdf",
            bbox_extra_artists=(legend,),
            bbox_inches="tight",
            facecolor="white",
        )
    else:
        # ax.set_title(title_map[method], fontsize=25)
        plt.savefig(
            out_path + f"{file_pref}-{method}-os.pdf",
            bbox_inches="tight",
            facecolor="white",
        )        

    return

################################################################################
#
# Multi-Objective Optimization Solver
#
################################################################################

# helper function for projection (adopted from LibMTL: https://github.com/median-research-group/LibMTL.git)
def projection2simplex(y):
    y = y.view(-1)
    m = len(y)
    sorted_y = torch.sort(y, descending=True)[0]
    tmpsum = 0.0
    tmax_f = (torch.sum(y) - 1.0)/m
    for i in range(m-1):
        tmpsum+= sorted_y[i]
        tmax = (tmpsum - 1)/ (i+1.0)
        if tmax > sorted_y[i+1]:
            tmax_f = tmax
            break
    return torch.max(y - tmax_f, torch.zeros(m).to(y.device))

def mean_grad(grads):
    return grads.mean(1)

def pcgrad(grads):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()
    if g12 < 0:
        return ((1-g12/g11)*g1+(1-g12/g22)*g2)/2
    else:
        return (g1+g2)/2

def mgd(grads):
    g1 = grads[:,0]
    g2 = grads[:,1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    if g12 < min(g11, g22):
        x = (g22-g12) / (g11+g22-2*g12 + 1e-8)
    elif g11 < g22:
        x = 1
    else:
        x = 0

    g_mgd = x * g1 + (1-x) * g2 # mgd gradient g_mgd
    return g_mgd

def moco(grads, y, lambd, beta, gamma, rho):

    # y update
    y = y - beta * ( y - grads )

    # lambda update
    lambd = projection2simplex( lambd - gamma * ( torch.transpose(y, 0, 1) @ ( y @ lambd ) + rho * lambd ) ).view([2, 1])
    
    g_moco = y@lambd 

    return g_moco.view(-1), y, lambd


def modo(grads1, grads2, lambd, gamma, rho):

    # lambda update
    lambd = projection2simplex( lambd - gamma * ( torch.transpose(grads1, 0, 1) @ ( grads2 @ lambd ) + rho * lambd ) ).view([2, 1])
    
    g_modo = 0.5 * (grads1 + grads2) @lambd 

    return g_modo.view(-1), lambd

# legacy func ----------------------------------------------------------------------------------
def tracking(grads, y1, y2, beta, sigma, it):
    g1 = grads[:,0]
    g2 = grads[:,1]

    y1_ = y1 - beta/(it+1)**sigma*(y1-g1)
    y2_ = y2 - beta/(it+1)**sigma*(y2-g2)

    y11 = y1_.dot(y1_).item()
    y12 = y1_.dot(y2_).item()
    y22 = y2_.dot(y2_).item()

    if y12 < min(y11, y22):
        x = (y22-y12) / (y11+y22-2*y12 + 1e-8)
    elif y11 < y22:
        x = 1
    else:
        x = 0
    
    g_tracking = x * y1_ + (1-x) * y2_ # tracking gradient g_tracking
    # print(y1_, y2_, g_tracking)
    return g_tracking, y1_, y2_, x, 1-x

def local_tracking(x, gamma, J):
    x_ = torch.zeros([2, 2])
    x_.requires_grad=True
    for m in range(2):
        x_[:, m] = x.detach().clone()        
        for j in range(J):
            _, g_ = F(x_[:, m], True, decay=0.0, it=0, sig=1, noise=True)
            x_[:, m] = x_[:, m] - gamma*g_[:,m]
    
    y1_ = -(x_[:,0].detach()-x.detach())/gamma
    y2_ = -(x_[:,1].detach()-x.detach())/gamma

    y11 = y1_.dot(y1_).item()
    y12 = y1_.dot(y2_).item()
    y22 = y2_.dot(y2_).item()

    if y12 < min(y11, y22):
        z = (y22-y12) / (y11+y22-2*y12 + 1e-8)
    elif y11 < y22:
        z = 1
    else:
        z = 0
    
    g_local_tracking = z * y1_ + (1-z) * y2_ 
    # print(y1_, y2_, z, 1-z, g_local_tracking)
    return g_local_tracking
# -----------------------------------------------------------------------------------------------------

def cagrad(grads, c=0.5):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g0 = (g1+g2)/2

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12+1e-4)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm+1e-4)
    g = g0 + lmbda * gw
    return g / (1+c)


def grad_comp():
    lr=0.01 #0.001
    sigma1=0.05 #0.05
    grads = {}
    all_traj_error_norm = {}

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
    ]

    methods = ["mgd", "smgd", "pcgrad", "cagrad", "tracking"]

    for i, init in enumerate(inits):
        for m in methods: #["mgd", "pcgrad", "cagrad", "tracking"]
            grads[m] = {"x":[],"grad_stoch":[], "grad_true":[], "multi_grad_stoch":[], "multi_grad_true":[]}
        
        x = init.clone()
        x.requires_grad = True

        n_iter = 70000 # 100000
        opt = torch.optim.Adam([x], lr=lr) # original
        # opt = torch.optim.SGD([x], lr=lr)
        decay = lambda epoch: 1/(epoch+1)**sigma1
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=decay)

        for it in range(n_iter):

            # traj.append(x.detach().numpy().copy())  
            f, grads_true = F(x, True, decay=0.0, it=it, sig=1, noise=False)
            f, grads_stoch = F(x, True, decay=0.0, it=it, sig=1, noise=True)
            if it==0:
                y1 = torch.zeros(grads_true[:,0].shape)
                y2 = torch.zeros(grads_true[:,0].shape)
                beta = 5.0
                sigma = 0.5

            for m in methods:
                solver = maps[m]

                if m=="cagrad":
                    g = solver(grads_true, c=0.5)
                    gs = solver(grads_stoch, c=0.5)
                elif m=="tracking":
                    g = maps["mgd"](grads_true)
                    gs, y1, y2, lambd1, lambd2 = solver(grads_stoch, y1, y2, beta, sigma, it)
                else:
                    g = solver(grads_true)
                    gs = solver(grads_stoch)  
                grads[m]["x"].append(x.clone())
                grads[m]["multi_grad_stoch"].append(gs.clone())
                grads[m]["multi_grad_true"].append(g.clone())
                grads[m]["grad_stoch"].append(grads_stoch.clone())
                grads[m]["grad_true"].append(grads_true.clone())
            opt.zero_grad()
            x.grad = gs
            opt.step()
            scheduler.step()                  

    torch.save(grads, f"toy{i}_grads.pt")

def plot_grad_dir():
    
    g1 = torch.load(f"toy0_grads.pt")
    idx=50000


    plt.plot([0, g1["tracking"]["grad_stoch"][idx][:, 0][0]], [0, g1["tracking"]["grad_stoch"][idx][:, 0][1]], 'b')
    plt.plot([0, g1["tracking"]["grad_stoch"][idx][:, 1][0]], [0, g1["tracking"]["grad_stoch"][idx][:, 1][1]], 'b')
    plt.plot([0, g1["tracking"]["multi_grad_stoch"][idx][0]], [0, g1["tracking"]["multi_grad_stoch"][idx][1]], 'r')
    plt.plot([0, g1["tracking"]["multi_grad_true"][idx][0]], [0, g1["tracking"]["multi_grad_true"][idx][1]], 'g--')
    plt.plot([0, g1["tracking"]["grad_true"][idx][:, 0][0]], [0, g1["tracking"]["grad_true"][idx][:, 0][1]], 'k--')
    plt.plot([0, g1["tracking"]["grad_true"][idx][:, 1][0]], [0, g1["tracking"]["grad_true"][idx][:, 1][1]], 'k--')
    print(g1["tracking"]["multi_grad_stoch"][idx])
    plt.show()


    # for method in ["cagrad"]: #["mgd", "pcgrad", "cagrad", "tracking"]
    #     ranges = list(range(10, length, 1000))
    #     ranges.append(length-1)
    #     for t in tqdm(ranges):
    #         plot_contour(F,
    #                      task=0, # task == 0 means plot for both tasks
    #                      traj=[t1[method][:t],t2[method][:t],t3[method][:t]],
    #                      plotbar=(method == "tracking"),
    #                      name=f"./imgs/toy_{method}_{t}")



### Start experiments ###

def run_all(seeds=1, lr=0.01, img_dir='temp', n_iter=50000):
    if not os.path.exists(f'./imgs/{img_dir}'):
        os.makedirs(f'./imgs/{img_dir}')

    
    sigma1=0.05 # for learning rate decay
    batch_size=2 # to accomodate modo

    all_traj = {}
    all_traj_error_norm = {}
    all_traj_true_dir_norm = {}

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([-8.5, -5.]),
        torch.Tensor([9.,   9.]),
        torch.Tensor([0.,   0.]),
        torch.Tensor([10., -8.]),
    ]

    for i, init in enumerate(inits):
        print(f'\ninit:{init}\n')
        for m in tqdm(["modo"]): #["mgd", "smgd", "pcgrad", "cagrad", "moco"]
            all_traj[m] = []
            all_traj_error_norm[m] = []
            all_traj_true_dir_norm[m] = []

            for seed in range(seeds):
                traj = []
                traj_error_norm = []
                traj_true_dir_norm = []
                solver = maps[m]
                x = init.clone()
                x.requires_grad = True

                opt = torch.optim.Adam([x], lr=lr) # original
                # opt = torch.optim.SGD([x], lr=lr)
                # decay = lambda epoch: 1/(epoch+1)**sigma1
                # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=decay)

                for it in range(n_iter):
                    traj.append(x.detach().numpy().copy())
                    if m=="mgd":
                        f, grads = F(x, compute_grad=True, data_type='pop')  # x, compute_grad=False, data_type='pop', batch_size=1
                        g = solver(grads)
                    elif m== "cagrad":
                        _, grads = F(x, compute_grad=True, data_type='stoch', stoch_type='grad', batch_size=batch_size)
                        g = solver(grads, c=0.5)
                    elif m=="moco":
                        if it==0:
                            y = torch.zeros(grads.shape)
                            lambd = 0.5 * torch.ones([2,1])
                            beta = 0.1
                            gamma = 0.1
                            rho = 0.01
                        _, grads = F(x, compute_grad=True, data_type='stoch', stoch_type='grad', batch_size=batch_size)
                        g, y, lambd = solver(grads, y, lambd, beta, gamma, rho)
                    elif m=="modo":
                        if it==0:
                            lambd = 0.5 * torch.ones([2,1])
                            gamma = 0.01
                            rho = 0.0
                        _, grads1 = F(x, compute_grad=True, data_type='pop', stoch_type='grad', batch_size=batch_size//2)
                        _, grads2 = F(x, compute_grad=True, data_type='pop', stoch_type='grad', batch_size=batch_size//2)
                        # print(grads1 == grads2)
                        g, lambd = solver(grads1, grads2, lambd, gamma, rho)
                    else:
                        _, grads = F(x, compute_grad=True, data_type='pop', stoch_type='grad', batch_size=batch_size)
                        g = solver(grads)
                    # calculate true multi gradient
                    # if m=="smgd" or m=="tracking":
                    #     f, grads_true = F(x, True, decay=0.0, it=it, sig=1, noise=False)
                    #     g_true = maps["mgd"](grads_true)
                    #     error_norm = np.linalg.norm(g_true-g)   
                    #     traj_error_norm.append(error_norm)

                    # for true mgd dir. norm calc. This is an indicator of PS point
                    f, grads_true = F(x, compute_grad=True, data_type='pop')
                    g_true = maps["mgd"](grads_true)
                    g_true_norm = np.linalg.norm(g_true)
                    traj_true_dir_norm.append(g_true_norm)

                    # print(f'it: {it}, grad: {g}')

                    opt.zero_grad()
                    x.grad = g
                    opt.step()
                    # scheduler.step()

                all_traj[m].append(torch.tensor(traj))
                all_traj_error_norm[m].append(torch.tensor(traj_error_norm))
                all_traj_true_dir_norm[m].append(torch.tensor(traj_true_dir_norm))
        torch.save(all_traj, f"./imgs/{img_dir}/toy{i}-runs{seeds}.pt")
        torch.save(all_traj_error_norm, f"./imgs/{img_dir}/toy{i}-error_norm-runs{seeds}.pt")
        torch.save(all_traj_true_dir_norm, f"./imgs/{img_dir}/toy{i}-true_dir_norm-runs{seeds}.pt")

def plot_results(img_dir='temp'):
    if not os.path.exists(f'./imgs/{img_dir}'):
        os.makedirs(f'./imgs/{img_dir}')
    plot3d(F)
    plot_contour(F, 1, name=f"./imgs/{img_dir}/_toy_task_1")
    plot_contour(F, 2, name=f"./imgs/{img_dir}/_toy_task_2")
    t1 = torch.load(f"./imgs/{img_dir}/toy0-runs1.pt")
    t2 = torch.load(f"./imgs/{img_dir}/toy1-runs1.pt")
    t3 = torch.load(f"./imgs/{img_dir}/toy2-runs1.pt")
    # t4 = torch.load(f"./imgs/{img_dir}/toy3-samples10.pt")
    # t5 = torch.load(f"./imgs/{img_dir}/toy4-samples10.pt")

    # length = t1["sgd"].shape[0] #original
    key_list = list(t1.keys())
    print('\n Loaded keys:\n')
    print(key_list)
    print()
    length = t1[key_list[0]][0].shape[0]
    # tt1 = torch.FloatTensor(t1["cagrad"]).mean(axis=0)
    
    # print(tt1.shape)

    for method in key_list: #
        ranges = list(range(10, length, 1000))
        ranges.append(length-1)
        for t in tqdm(ranges):
            plot_contour(F,
                         task=0, # task == 0 means plot for both tasks
                        #  traj=[np.mean(np.array(t1[method]), axis=0)[:t],np.mean(np.array(t2[method]), axis=0)[:t],np.mean(np.array(t3[method]), axis=0)[:t]],
                        traj=[t1[method][0][:t],t2[method][0][:t],t3[method][0][:t]], #,t4[method][0][:t],t5[method][0][:t]
                        # traj=[t4[method][0][:t],t5[method][0][:t]],
                         plotbar=(method == "tracking"),
                         name=f"./imgs/{img_dir}/toy_{method}_{t}")


if __name__ == "__main__":
    # parameters
    lr=0.001
    gamma=0.0001
    noise_sigma=0.0
    n_iter=200000
    pref=f'modo_pf_n_iter-200000_gamm-{gamma}'

    # Define the problem
    F = Toy(grad_sig=noise_sigma)

    # input grad manipulation methods
    maps = {
        "sgd": mean_grad,
        "cagrad": cagrad,
        "mgd": mgd,
        "smgd":mgd,
        "pcgrad": pcgrad,
        "moco": moco,
        "modo":modo
    }

    img_dir=f"{pref}_noise_sigma-{noise_sigma}_lr-{lr}"
    # print(f'\n{img_dir}\n')


    # For running the toy example and generate trajectories
    # run_all(lr=lr, img_dir=img_dir, n_iter=n_iter) 

    # # plot 3d mean error surface
    # plot3d(F, data_type='emp')

    # plot 2d contour
    # plot_contour(F, task=0, name='Loss 1')
    
    # Plot trajectories    
    # plot_results(img_dir=img_dir)    

    # Plot Pareto trajectory in Pareto front
    plot_2d_pareto("modo", out_path=f"imgs/{img_dir}/")

    # plot 3d PS measure surface
    # plot3d_PS(F)   
	
    # t = torch.load(f"toy2.pt")   
    ## Plot loss landscape        
    # plotme(F, all_traj=t, xl=11)
    # plot_loss(t, name="L2vsK_traj2")
    # plot_contour(F, task=0)

    ## Plot Pareto trajectory in Pareto front
    # plot_2d_pareto("tracking")
    # plot_2d_pareto("smgd")
    
    ## Other plots that were not used in paper
    # grad_comp()
    # plot_grad_dir()
