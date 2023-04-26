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

    def forward(self, x, compute_grad=False, decay=0, it=0, sig=4, noise=False, batch_size=1):
        x1 = x[0]
        x2 = x[1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6 # Original
        # f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 60
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20 # Original
        # f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 100
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
            if noise:
                for b in range(batch_size):
                    g = g + torch.normal(0, sig/(it+1)**decay, size=g.shape)
            return f, g
        else:
            return f

    def batch_forward(self, x):
        x1 = x[:,0]
        x2 = x[:,1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6 # Original
        # f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 60
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20 # Original
        # f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 100
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = ( f1 * c1 + f1_sq * c2 )*2
        f2 = f2 * c1 + f2_sq * c2

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
        "tracking": "tab:olive",
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

def plot3d(F, xl=11):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.mean(1).view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=cm.viridis)
    
    zmin = Ys.mean(1).min()
    zmax = Ys.mean(1).max()

    # ax.set_zticks([-16, -8, 0, 8]) # Original
    # ax.set_zlim(-20, 10) # Original
    ax.set_zticks([zmin, zmin/2, 0, zmax/2, zmax])
    ax.set_zlim(zmin-1, zmax+1)

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
    plt.savefig(f"3d-obj.png", dpi=1000)

def plot_contour(F, task=1, traj=None, xl=11, plotbar=False, name="tmp"): 
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)

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
    if task == 0:
        Yv = Ys.mean(1)
        plt.plot(-8.5, 7.5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot(-8.5, -5, marker='o', markersize=10, zorder=5, color='k')
        plt.plot( 9, 9, marker='o', markersize=10, zorder=5, color='k')
        plt.plot([-7, 7], [yy, yy], linewidth=8.0, zorder=0, color='gray')
        # plt.plot(0, yy, marker='*', markersize=15, zorder=5, color='k') # Original
        plt.plot(xx, yy, marker='*', markersize=15, zorder=5, color='k')
    elif task == 1:
        Yv = Ys[:,0]
        plt.plot(7, yy, marker='*', markersize=15, zorder=5, color='k')
    else:
        Yv = Ys[:,1]
        plt.plot(-7, yy, marker='*', markersize=15, zorder=5, color='k')

    c = plt.contour(X, Y, Yv.view(n,n), cmap=cm.viridis, linewidths=4.0)

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

    t1 = torch.load(f"toy0-samples10.pt")
    t2 = torch.load(f"toy1-samples10.pt")
    t3 = torch.load(f"toy2-samples10.pt")
    t4 = torch.load(f"toy3-samples10.pt")
    t5 = torch.load(f"toy4-samples10.pt")

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
    if method == "mgd":
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

    if method != "mgd":
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    if method == "tracking":
        legend = ax.legend(
            loc=2, bbox_to_anchor=(-0.15, 1.3), frameon=False, fontsize=20, ncol=2
        )

        # ax.set_title(title_map[method], fontsize=25)
        plt.savefig(
            out_path + f"{method}-os.png",
            bbox_extra_artists=(legend,),
            bbox_inches="tight",
            facecolor="white",
        )
    else:
        # ax.set_title(title_map[method], fontsize=25)
        plt.savefig(
            out_path + f"{method}-os.png",
            bbox_inches="tight",
            facecolor="white",
        )        

    return

################################################################################
#
# Multi-Objective Optimization Solver
#
################################################################################

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
    lr=0.001 #0.001
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
    "tracking": tracking,
    "local-tracking": local_tracking,
    "smgd-minibatch":mgd,
}

### Start experiments ###

def run_all(seeds=10):
    lr=0.001 #0.001
    sigma1=0.05 #0.05
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
        for m in tqdm(["tracking"]): #["mgd", "smgd", "pcgrad", "cagrad", "tracking"]
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

                n_iter = 70000 # 100000
                opt = torch.optim.Adam([x], lr=lr) # original
                # opt = torch.optim.SGD([x], lr=lr)
                decay = lambda epoch: 1/(epoch+1)**sigma1
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=decay)


                for it in range(n_iter):
                    traj.append(x.detach().numpy().copy())  

                    if m=="mgd":
                        f, grads = F(x, True, decay=0.0, it=it, sig=1, noise=False)
                    else:
                        f, grads = F(x, True, decay=0.0, it=it, sig=1, noise=True)


                    if m== "cagrad":
                        g = solver(grads, c=0.5)
                    elif m=="tracking":
                        if it==0:
                            y1 = torch.zeros(grads[:,0].shape)
                            y2 = torch.zeros(grads[:,0].shape)
                            beta = 5.0
                            sigma = 0.5
                        g, y1, y2, lambd1, lambd2 = solver(grads, y1, y2, beta, sigma, it)
                    elif m=="local-tracking":
                        J=4
                        gamma=0.1
                        g = solver(x, gamma, J)

                    else:
                        g = solver(grads)
                    # calculate true multi gradient
                    # if m=="smgd" or m=="tracking":
                    #     f, grads_true = F(x, True, decay=0.0, it=it, sig=1, noise=False)
                    #     g_true = maps["mgd"](grads_true)
                    #     error_norm = np.linalg.norm(g_true-g)   
                    #     traj_error_norm.append(error_norm)

                    # for true mgd dir. norm calc. This is an indicator of PS point
                    f, grads_true = F(x, True, decay=0.0, it=it, sig=1, noise=False)
                    g_true = maps["mgd"](grads_true)
                    g_true_norm = np.linalg.norm(g_true)
                    traj_true_dir_norm.append(g_true_norm)
                    if it>60000 and it%100==0:
                        print()
                        print(it)
                        print("true grads", grads_true)
                        print("grads", grads)
                        print("g", g)
                        print("y1", y1)
                        print("y2", y2)
                        print("lambda", lambd1, lambd2)


                    opt.zero_grad()
                    x.grad = g
                    opt.step()
                    scheduler.step()

                all_traj[m].append(torch.tensor(traj))
                all_traj_error_norm[m].append(torch.tensor(traj_error_norm))
                all_traj_true_dir_norm[m].append(torch.tensor(traj_true_dir_norm))
        print(all_traj)
        torch.save(all_traj, f"toy{i}-runs{seeds}.pt")
        torch.save(all_traj_error_norm, f"toy{i}-error_norm-runs{seeds}.pt")
        torch.save(all_traj_true_dir_norm, f"toy{i}-true_dir_norm-runs{seeds}.pt")


def run_mul_grad_error(samples=10, runs=5):
    lr=0.005 #0.001
    sigma1=0.075 #0.05
    all_traj = {}
    all_traj_error_norm = {}

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([-8.5, -5.]),
        torch.Tensor([9.,   9.]),
        # torch.Tensor([0.,   0.]),
        # torch.Tensor([10., -8.]),
    ]

    for i, init in enumerate(inits):
        print("init", i)
        for m in tqdm(["smgd", "smgd-minibatch", "tracking"]): #["mgd", "pcgrad", "cagrad", "tracking"]
            print("method", m)
            all_traj[m] = []
            all_traj_error_norm[m] = []

            for run in range(runs):
                print("run", run)
                traj = []
                traj_error_norm = []
                solver = maps[m]
                x = init.clone()
                x.requires_grad = True

                n_iter = 70000 # 100000
                opt = torch.optim.Adam([x], lr=lr) # original
                # opt = torch.optim.SGD([x], lr=lr)
                decay = lambda epoch: 1/(epoch+1)**sigma1
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=decay)


                for it in range(n_iter):
                    traj.append(x.detach().numpy().copy())  

                    if m=="mgd":
                        f, grads = F(x, True, decay=0.0, it=it, sig=1, noise=False)
                        g = solver(grads)
                    else:
                        g_=0
                        for sample in range(samples):
                            f, grads = F(x, True, decay=0.0, it=it, sig=1, noise=True)

                            if m=="tracking":
                                if it==0:
                                    y1 = torch.zeros(grads[:,0].shape)
                                    y2 = torch.zeros(grads[:,0].shape)
                                    beta = 5.0
                                    sigma = 0.5
                                if sample==samples-1:
                                    g, y1, y2, lambd1, lambd2 = solver(grads, y1, y2, beta, sigma, it)
                                else:
                                    g, _, _, _, _ = solver(grads, y1, y2, beta, sigma, it)
                            elif m=="local-tracking":
                                J=4
                                gamma=0.1
                                g = solver(x, gamma, J)
                            elif m=="smgd-minibatch":
                                smgd_samples= it//10000+1
                                g_smgd = 0
                                for smgd_sample in range(smgd_samples):
                                    f, grads = F(x, True, decay=0.0, it=it, sig=1, noise=True)
                                    g_smgd += grads
                                g = solver(g_smgd/smgd_samples)
                            else:
                                g = solver(grads)
                            
                            g_ += g.detach().numpy().copy()


                    # if it%100==0:
                    # #     print(f"trj: {i}, it: {it}, grad: {g.detach().numpy()}, x:{x.detach().numpy()}")
                    #     print(f"trj: {i}, it: {it}, error_norm: {error_norm}, x:{x.detach().numpy()}")
                    #     # print(f"trj: {i}, it: {it}, f: {f}, x:{x.detach().numpy()}, g:{g.detach().numpy()}, lambda:{lambd1, lambd2}")
                            # calculate true multi gradient
                        if m=="smgd" or m=="tracking" or m=="smgd-minibatch":
                            f, grads_true = F(x, True, decay=0.0, it=it, sig=1, noise=False)
                            g_true = maps["mgd"](grads_true)
                            error_norm = np.linalg.norm(g_true-g_/samples)   
                            traj_error_norm.append(error_norm)

                    opt.zero_grad()
                    x.grad = g
                    opt.step()
                    scheduler.step()

                all_traj[m].append(torch.tensor(traj))
                all_traj_error_norm[m].append(torch.tensor(traj_error_norm))
        # print(all_traj)
        torch.save(all_traj, f"toy{i}-samples{samples}.pt")
        torch.save(all_traj_error_norm, f"toy{i}-error_norm-samples{samples}.pt")


def plot_results():
    plot3d(F)
    plot_contour(F, 1, name="toy_task_1")
    plot_contour(F, 2, name="toy_task_2")
    t1 = torch.load(f"toy0-runs10.pt")
    t2 = torch.load(f"toy1-runs10.pt")
    t3 = torch.load(f"toy2-runs10.pt")
    # t4 = torch.load(f"toy3-samples10.pt")
    # t5 = torch.load(f"toy4-samples10.pt")

    # length = t1["sgd"].shape[0] #original
    length = t1["smgd"][0].shape[0]
    # tt1 = torch.FloatTensor(t1["cagrad"]).mean(axis=0)
    
    # print(tt1.shape)

    for method in ["mgd", "smgd", "pcgrad", "cagrad", "tracking"]: #
        ranges = list(range(10, length, 1000))
        ranges.append(length-1)
        for t in tqdm(ranges):
            plot_contour(F,
                         task=0, # task == 0 means plot for both tasks
                        #  traj=[np.mean(np.array(t1[method]), axis=0)[:t],np.mean(np.array(t2[method]), axis=0)[:t],np.mean(np.array(t3[method]), axis=0)[:t]],
                        traj=[t1[method][0][:t],t2[method][0][:t],t3[method][0][:t]], #,t4[method][0][:t],t5[method][0][:t]
                        # traj=[t4[method][0][:t],t5[method][0][:t]],
                         plotbar=(method == "tracking"),
                         name=f"./imgs/toy_{method}_{t}")


if __name__ == "__main__":
    # For running the toy example and generate trajectories
    run_all() 
    
    ## Plot trajectories    
    # plot_results()    	
    # t = torch.load(f"toy2.pt")   
    ## Plot loss landscape        
    # plotme(F, all_traj=t, xl=11)
    # plot_loss(t, name="L2vsK_traj2")
    # plot_contour(F, task=0)

    ## Plot error norm reduction with iterations and cumulative samples.
    # t_norm0 = torch.load(f"toy0-error_norm-samples10.pt")
    # N = t_norm0["smgd"][0].shape[0]
    # sam = [i//10000+1 for i in np.arange(N)]
    # cumsam = np.array([np.sum(np.array(sam)[:i+1]) for i in np.arange(N)])[::100]
    # plot_error_norm(t_norm0, name="error_norm0", cum_sam=cumsam)
    # t_norm1 = torch.load(f"toy1-error_norm-samples10.pt") 
    # plot_error_norm(t_norm1, name="error_norm1", cum_sam=cumsam)
    # t_norm2 = torch.load(f"toy2-error_norm-samples10.pt") 
    # plot_error_norm(t_norm2, name="error_norm2", cum_sam=cumsam)
    # t_norm3 = torch.load(f"toy3-error_norm-samples10.pt") 
    # plot_error_norm(t_norm3, name="error_norm3")
    # t_norm4 = torch.load(f"toy4-error_norm-samples10.pt") 
    # plot_error_norm(t_norm4, name="error_norm4")

    ## Plot Pareto trajectory in Pareto front
    # plot_2d_pareto("tracking")
    # plot_2d_pareto("smgd")
    
    ## Other plots that did not used in paper
    # grad_comp()
    # plot_grad_dir()

    ## Similar to run with lookahead method
    # run_mul_grad_error()

    # # Plot true mgda direction norm with iterations.
    # t_norm0 = torch.load(f"toy0-true_dir_norm-runs10.pt")
    # plot_error_norm(t_norm0, name="true_dir_norm0")
    # t_norm1 = torch.load(f"toy1-true_dir_norm-runs10.pt") 
    # plot_error_norm(t_norm1, name="true_dir_norm1")
    # t_norm2 = torch.load(f"toy2-true_dir_norm-runs10.pt") 
    # plot_error_norm(t_norm2, name="true_dir_norm2")
    # t_norm3 = torch.load(f"toy3-true_dir_norm-runs10.pt") 
    # plot_error_norm(t_norm3, name="true_dir_norm3")
    # t_norm4 = torch.load(f"toy4-true_dir_norm-runs10") 
    # plot_error_norm(t_norm4, name="error_norm4") 
