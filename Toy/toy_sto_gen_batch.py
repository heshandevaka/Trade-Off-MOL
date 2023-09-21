"""Adaptation of code from: https://github.com/Cranial-XIX/CAGrad"""

from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker, rc, font_manager
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

import scipy
import scipy.io

################################################################################
#
# Define the Optimization Problem
#
################################################################################
LOWER = 0.000005

class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([
            [-3.0, 0],
            [3.0, 0]])
        
        self.num_data = 20 # number of empirical data 
        self.data_mean = 0 # mean of population data distribution
        self.data_sig = 1. # std of population distribution
        data = scipy.io.loadmat('./data/synthetic_NC_data.mat')
        z1 = data['z1'].reshape(-1)
        z2 = data['z2'].reshape(-1)
        self.emp_data_set = torch.stack([torch.from_numpy(z1), 
                                        torch.from_numpy(z2)])
        # self.emp_data_set = torch.normal(
        #     self.data_mean, self.data_sig, size=(self.num_data, 2)).detach()
        self.emp_data_mean = torch.mean(self.emp_data_set, dim=1).view([2, 1])

        self.p1 = 3.5
        self.p2 = -3.5
        self.p3 = -1

    def __forward__(self, x1, x2, z=[0,0], compute_grad=False): 
        
        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6 
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh((x2)*0.5), 0)
        c2 = torch.clamp(torch.tanh(-(x2)*0.5), 0)

        f1_sq = ((-x1+self.p1).pow(2) + 0.5*(-x2+self.p3).pow(2)) / 10 - 20 
        f2_sq = ((-x1+self.p2).pow(2) + 0.5*(-x2+self.p3).pow(2)) / 10 - 20 

        f1 = f1 * c1 + (f1_sq - 2*z[0] * x1 - 5.5* z[1] * x2) * c2
        f2 = f2 * c1 + (f2_sq + 2*z[0] * x1 - 5.5* z[1] * x2) * c2 

        f  = torch.cat([f1.view(-1, 1), f2.view(-1,1)], -1)
        if compute_grad:
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            g = torch.Tensor([[g11, g21], [g12, g22]])
            return f, g
        else:
            return f

    def forward(self, x, compute_grad=False, data_type='pop', batch_size=1): 
        # data types: 'stoch', 'emp', 'pop'
        x1 = x[0]
        x2 = x[1]
        
        if data_type=='pop':
            z=[0,0]
        if data_type=='stoch':
            batch_idx = np.random.choice(self.num_data, batch_size)
            z = self.emp_data_set[:,batch_idx] # 2*batch_size
            z_m = torch.mean(z, dim=1).view([2, 1])
            z=z_m
        if data_type=='emp':
            z=self.emp_data_mean
        
        if compute_grad:
            f, g = self.__forward__(x1, x2, z=z, compute_grad=True) 
            return f, g
        else:
            f = self.__forward__(x1, x2, z=z, compute_grad=False) 
            return f

    def batch_forward(self, x, 
                    data_type='pop', compute_grad=False): 
        # data types: 'emp', 'pop'

        if compute_grad:
            g = []
            g_emp = []
            f = []
            for i, x_ in enumerate(x):

                f_, g_ = self.forward(x_, compute_grad=True,
                                data_type='pop', batch_size=1)
                g.append(g_.clone())
                f.append(f_.clone())
                f_emp, g_emp_ = self.forward(x_, compute_grad=True,
                                            data_type='emp', batch_size=1)
                g_emp.append(g_emp_.clone())
            
            return torch.stack(f), torch.stack(g), torch.stack(g_emp)
        
        else:
            x1 = x[:,0]
            x2 = x[:,1]

            if data_type == 'pop':
                z=[0,0]
            # for plotting emperical objective
            elif data_type == 'emp':
                print('torch.sum( x * self.emp_data_mean.view(-1), dim=1)', 
                      torch.sum( x * self.emp_data_mean.view(-1), dim=1).shape)
                z = self.emp_data_mean
            
            f = self.__forward__(x1, x2, z=z, compute_grad=False)
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
    c = plt.contour(X, Y, PS, cmap=cm.viridis, 
                    linewidths=4.0, linestyles='dotted')
    c1 = plt.contour(X, Y, PS_emp, cmap=cm.viridis, linewidths=4.0)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.yticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.tight_layout()
    plt.savefig(f"./imgs/_2d-PS.png", dpi=100)
    plt.close()


def plot_contour(F, emp=True, task=1, traj=None, 
                 xl=11, levels=12, plotbar=False, name="tmp"): 
    # 
    # rc('font',**{'family':'serif','serif':['Times']})
    font_manager.findfont("Times New Roman")
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    tick_fontsize = 22
    label_fontsize = 28
    
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    if task == 2:
        fig = plt.figure(figsize=(7,6))
    else:
        if plotbar:
            fig = plt.figure(figsize=(7,6))
        else:    
            fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    Ys_emp = F.batch_forward(Xs, data_type='emp')

    cmap = cm.get_cmap('viridis')
    
    Ys1 = Ys[:,0].numpy()
    meshy1 = np.argmin(Ys1) // n
    meshx1 = np.argmin(Ys1) % n
    yy1 = y[meshy1]
    xx1 = x[meshx1]

   
    Ys2 = Ys[:,1].numpy()
    meshy2 = np.argmin(Ys2)//n
    meshx2 = np.argmin(Ys2)%n
    yy2 = y[meshy2]
    xx2 = x[meshx2]

    # Added this block to remove hardcoded calcs
    Ysmean = Ys.mean(1).numpy()
    Ysmin = np.min(Ysmean)
    Ysargmin = np.argmin(Ysmean)
    meshy = np.argmin(Ysmean)//n
    meshx = np.argmin(Ysmean)%n
    

    yy = y[meshy]
    xx = x[meshx]

    # plot mean objective
    if task == 0:
        # get mean of objectives
        Yv = Ys.mean(1)
        Yv_emp = Ys_emp.mean(1)
        # intial points
        plt.plot(-8.5, 7.5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot(-10, -3, marker='o', markersize=10, zorder=5, color='k')
        plt.plot( 9, 9, marker='o', markersize=10, zorder=5, color='k')
        # pareto front
        plt.plot([xx1, xx2], [yy1, yy2], linewidth=8.0, zorder=0, 
                color='green', alpha=0.5)
        # optimum of mean loss
        
        # Original
        plt.plot(xx, yy, marker='*', markersize=15, zorder=5, color='g')
        c = plt.contour(X, Y, Yv.view(n,n), cmap=cm.viridis, 
                        linewidths=3., levels=levels, 
                        linestyles='dotted')
        cf = plt.contourf(X, Y, Yv.view(n,n), cmap=cm.viridis, 
                        levels=levels, alpha=0.3, 
                        linewidths=0.5, 
                        linestyle='dotted')
    # plot objective 1
    elif task == 1:
        # get first objective values
        Yv_emp = Ys_emp[:, 0]
        # optimum of loss 1
        plt.plot(xx1, yy1, marker='*', markersize=15, zorder=5, color='g')
        c = plt.contour(X, Y, Ys[:,0].view(n,n), cmap=cm.viridis, 
                        linewidths=3., levels=levels, 
                        linestyles='dotted')
        cf = plt.contourf(X, Y, Ys[:,0].view(n,n), cmap=cm.viridis, 
                        levels=levels, alpha=0.3, 
                        linewidths=0.5, linestyles='dotted')
        
    # plot objective 2
    elif task == 2:
        # get second objective values
        Yv_emp = Ys_emp[:, 1]
        # optimum of loss2
        plt.plot(xx2, yy2, marker='*', markersize=15, zorder=5, color='g')
        c = plt.contour(X, Y, Ys[:,1].view(n,n), cmap=cm.viridis, 
                        linewidths=3., levels=levels, 
                        linestyles='dotted')
        cf = plt.contourf(X, Y, Ys[:,1].view(n,n), cmap=cm.viridis, 
                        levels=levels, alpha=0.3, 
                        linewidths=0.5, 
                        linestyles='dotted')
        
        plt.ylabel(r"$x_2$", fontsize=label_fontsize)

    
    if emp:
        c_emp = plt.contour(X, Y, Yv_emp.view(n,n), cmap=cm.viridis, 
                            levels=levels, linewidths=4.0)
        
        Ys1 = Ys_emp[:,0].numpy()
        meshy1 = np.argmin(Ys1)//n
        meshx1 = np.argmin(Ys1)%n
        yy1 = y[meshy1]
        xx1 = x[meshx1]

        Ys2 = Ys_emp[:,1].numpy()
        meshy2 = np.argmin(Ys2)//n
        meshx2 = np.argmin(Ys2)%n
        yy2 = y[meshy2]
        xx2 = x[meshx2]

        meshy = np.argmin(Yv_emp)//n
        meshx = np.argmin(Yv_emp)%n
        yy = y[meshy]
        xx = x[meshx]

        if task == 0:
            plt.plot([xx1, xx2], [yy1, yy2], 
                     linewidth=8.0, zorder=0, color='gray')
        plt.plot(xx, yy, marker='*', markersize=15, zorder=5, color='k')

    if traj is not None:
        for tt in traj:
            l = tt.shape[0]
            color_list = np.zeros((l,3))
            color_list[:,0] = 1.
            color_list[:,1] = np.linspace(0, 1, l)
            
            ax.scatter(tt[:,0], tt[:,1], color=color_list, s=6, zorder=10)

    if plotbar:
        cbar = fig.colorbar(c, ticks=[ -18, -15, -13, -10, -5, 0, 3, 5])
        cbar.ax.tick_params(labelsize=tick_fontsize)
        

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=tick_fontsize)
    plt.yticks([-10, -5, 0, 5, 10], fontsize=tick_fontsize)
    plt.xlabel(r"$x_1$", fontsize=label_fontsize)
    plt.ylabel(r"$x_2$", fontsize=label_fontsize)
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
    for method in [ "mgd", "smgd", "pcgrad", "cagrad", "tracking"]: 
        # ["sgd", "mgd", "pcgrad", "cagrad", "tracking"], ["smgd", "smgd-minibatch", "tracking"]

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
def plot_2d_pareto(method, out_path="", data_type='pop'):
    sns.set_style("darkgrid", {"grid.linewidth": 1, "grid.color": "1",
                               'axes.facecolor':'lightsteelblue'})
    font_manager.findfont("Times New Roman")
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    tick_fontsize = 22
    label_fontsize = 28
    
    
    if method=="modo":
        folder_name = "results/modo/"
    elif method=="sgd":
        folder_name = "results/static/"
    elif method=="MGDA":
        folder_name = "results/MGDA/"
    
    t1 = torch.load(folder_name + f"toy0-runs1.pt")
    t2 = torch.load(folder_name + f"toy1-runs1.pt")
    t3 = torch.load(folder_name + f"toy2-runs1.pt")
    
    
    trajectories = {1:t1, 2:t2, 3:t3}
    fig, ax = plt.subplots(figsize=(6, 5))
    

    F = Toy()

    losses = []
    for res in trajectories.values():
        losses.append(F.batch_forward(res[method][0], data_type=data_type)) 
        # losses.append(F.batch_forward(torch.from_numpy(res[method]))) 
        
    n = 1000
    xl = 11
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs, data_type=data_type)
    
    Ys1 = Ys[:,0].numpy()
    meshy1 = np.argmin(Ys1) // n
    meshx1 = np.argmin(Ys1) % n
    yy1 = y[meshy1]
    xx1 = x[meshx1]

    Ys2 = Ys[:,1].numpy()
    meshy2 = np.argmin(Ys2)//n
    meshx2 = np.argmin(Ys2)%n
    yy2 = y[meshy2]
    xx2 = x[meshx2]

    x = np.linspace(xx1, xx2, 200)
    y = np.linspace(yy1, yy2, 200)

    inpt = np.stack((x, y)).T
    Xps = torch.from_numpy(inpt).double()

    Yps = F.batch_forward(Xps, data_type=data_type)
    if data_type=='emp':
        color_PF = "#72727A"
        alpha_PF = 1
        label_PF = "Empirical Pareto Front"
    elif data_type=='pop':
        color_PF = "g"
        alpha_PF = 0.9
        label_PF = "Population Pareto Front"
    
    ax.plot(
        Yps.numpy()[:, 0],
        Yps.numpy()[:, 1],
        "-",
        linewidth=8,
        color=color_PF,
        label=label_PF,
        alpha=alpha_PF
    )  # Pareto front
    
    count_s=0
    for i, tt in enumerate(losses):
        # print(tt[0, 0])
        count_s+=1
        ax.scatter(
            tt[0, 0], tt[0, 1],
            color="k",
            s=150,
            zorder=10,
            label="Initial Iterate" if i == 0 else None,
        )
        ttt = tt[0:50000]
        if i==0:
            colors = matplotlib.cm.viridis(np.linspace(0.0, 1, ttt.shape[0]))
        elif i==1:
            colors = matplotlib.cm.plasma_r(np.linspace(1, 0., ttt.shape[0]))
        elif i==2:
            colors = matplotlib.cm.autumn(np.linspace(0.0, 1, ttt.shape[0]))
        
        # print(tt.shape)
        ax.scatter(ttt[:, 0], ttt[:, 1], color=colors, s=5, zorder=9)
        ax.scatter(ttt[-1, 0], ttt[-1, 1], color='yellow', s=150, 
                   alpha=0.7, zorder=10,
                   label="Last Iterate" if i == 0 else None,)
        

    sns.despine()
    if data_type =='pop':
        ax.set_xlabel(r"$f_1$", size=label_fontsize)
        ax.set_ylabel(r"$f_2$", size=label_fontsize)
    elif data_type =='emp':
        ax.set_xlabel(r"$f_{S,1}$", size=label_fontsize)
        ax.set_ylabel(r"$f_{S,2}$", size=label_fontsize)


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

    
    
    if method == "modo":
        legend = ax.legend(
            loc=2, bbox_to_anchor=(0.2, 0.65), 
            frameon=True, fontsize=20, 
            framealpha=0.5,
        )
        legend.set_zorder(10)
        legend.get_frame().set_edgecolor('k')
        legend.get_frame().set_linewidth(1.0)
        ax.set_zorder(1)
    
        
    plt.savefig(
        out_path + f"{method}-{data_type}-os.png",
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

### Define the problem ###
F = Toy()

maps = {
    "sgd": mean_grad,
    "cagrad": cagrad,
    "mgd": mgd,
    "smgd":mgd,
    "pcgrad": pcgrad,
    "moco": moco,
    "modo":modo
}

### Start experiments ###

def run_all(seeds=1):
    lr = 0.005 # this is the step size alpha
    sigma1 = 0.05 #0.05
    batch_size = 16
    all_traj = {}
    all_traj_error_norm = {}
    all_traj_true_dir_norm = {}

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([-10, -3.]),
        # torch.Tensor([-10, -10.]),
        # torch.Tensor([-8.5, -5.]),
        torch.Tensor([9.,   9.]),
    ]

    for i, init in enumerate(inits):
        print(f'\ninit:{init}\n')
        for m in tqdm(["modo"]):
        # for m in tqdm(["sgd"]):
        # for m in tqdm(["mgd"]):
        # "sgd" is for Static and "mgd" is for MGDA
             
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

                n_iter = 50000 # this is #iterations T
                opt = torch.optim.Adam([x], lr=lr) # original
                # opt = torch.optim.SGD([x], lr=lr)
                decay = lambda epoch: 1/(epoch+1)**sigma1
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt, lr_lambda=decay)

                for it in range(n_iter):
                    traj.append(x.detach().numpy().copy())  

                    if m=="mgd":
                        f, grads = F(x, True, 'emp')  
                        # x, compute_grad=False, data_type='pop', batch_size=1
                        g = solver(grads)
                    elif m== "cagrad":
                        _, grads = F(x, True, 'stoch', batch_size)
                        g = solver(grads, c=0.5)
                    
                    elif m=="modo":
                        if it==0:
                            lambd = 0.5 * torch.ones([2,1]) # init lambda
                            gamma = 1e-4  # 8e-2 # this is step size gamma
                            rho = 0 #1e-16
                        _, grads1 = F(x, True, 'stoch', batch_size//2)
                        _, grads2 = F(x, True, 'stoch', batch_size//2)
                        g, lambd = solver(grads1, grads2, lambd, gamma, rho)
                    elif m=="local-tracking":
                        J=4
                        gamma=0.1
                        g = solver(x, gamma, J)

                    else:
                        _, grads = F(x, True, 'stoch', batch_size)
                        g = solver(grads)
                    
                    f, grads_true = F(x, True, 'emp')
                    g_true = maps["mgd"](grads_true)
                    g_true_norm = np.linalg.norm(g_true)
                    traj_true_dir_norm.append(g_true_norm)

                    if it%500 == 0:
                        print(f'it: {it}, grad: {g}')

                    opt.zero_grad()
                    x.grad = g
                    opt.step()
                    scheduler.step()

                all_traj[m].append(torch.tensor(traj))
                all_traj_error_norm[m].append(torch.tensor(traj_error_norm))
                all_traj_true_dir_norm[m].append(
                    torch.tensor(traj_true_dir_norm))
        
        
        # folder_name = "results/gamma/1E-4/"
        folder_name = "results/modo/"
        # folder_name = "results/static/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        torch.save(all_traj, folder_name+f"toy{i}-runs{seeds}.pt")
        torch.save(all_traj_error_norm, 
                   folder_name+f"toy{i}-error_norm-runs{seeds}.pt")
        torch.save(
            all_traj_true_dir_norm, 
            folder_name+f"toy{i}-true_dir_norm-runs{seeds}.pt")

def plot_results():
    plot3d(F)
    levels = [-20, -18, -15, -13, -10, -5, 0, 3, 5, 10]
    plot_contour(F, task=1, levels=levels, name="./imgs/_toy_task_1")
    plot_contour(F, task=2, levels=levels, name="./imgs/_toy_task_2")
    
    folder_name = "results/modo/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    t1 = torch.load(folder_name+"toy0-runs1.pt")
    t2 = torch.load(folder_name+"toy1-runs1.pt")
    t3 = torch.load(folder_name+"toy2-runs1.pt")
    

    key_list = list(t1.keys())
    print('\n Loaded keys:\n')
    print(key_list)
    print()
    length = t1[key_list[0]][0].shape[0]
    
    for method in key_list: #
        ranges = list(range(10, length, 2000))
        ranges.append(length-1)
        for t in tqdm(ranges):
            
            plot_contour(F, task=0, levels=levels,
                    traj=[t1[method][0][:t],t2[method][0][:t],t3[method][0][:t]], #,t4[method][0][:t],t5[method][0][:t]
                    plotbar=(method == "modo"),
                    name=folder_name+f"imgs/toy_{method}_{t}")


if __name__ == "__main__":
    # For running the toy example and generate trajectories
    run_all() 

    # Plot trajectories    
    plot_results()    

   
    ## Plot Pareto trajectory in Pareto front
    # plot_2d_pareto("modo", 
    #                out_path='imgs/gamma1E-2_', data_type='pop')
    # plot_2d_pareto("modo", 
    #                out_path='imgs/gamma1E-2_', data_type='emp')
    

    plot_2d_pareto("modo", out_path='imgs/',  data_type='pop')
    plot_2d_pareto("modo", out_path='imgs/', data_type='emp')
    # plot_2d_pareto("mgd", out_path='imgs/', data_type='pop')
    # plot_2d_pareto("mgd", out_path='imgs/', data_type='emp')
    # plot_2d_pareto("sgd", out_path='imgs/', data_type='pop')
    # plot_2d_pareto("sgd", out_path='imgs/', data_type='emp')
    
    

