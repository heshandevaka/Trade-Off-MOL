import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class PW(AbsWeighting):
    r"""Preference Weighting (PW).

    The loss weight for each task is some constant ``lambda`` in every iteration, where ``lambda`` denotes a vector in the simplex.

    """
    def __init__(self):
        super(PW, self).__init__()
        
    def backward(self, losses, **kwargs):
        # get preference weight
        lambd0 = kwargs['lambda']
        loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        loss.backward()
        return np.ones(self.task_num)