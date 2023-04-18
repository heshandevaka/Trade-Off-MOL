import torch, sys, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()
        # a flag to check whether shared param calculation was done once
        self._calc_shared_param = False
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        """Counts number of parameters at each layer of shared model (e.g. encoder),
            and returns the total.

        Returns:
        Total number of parameters in the shared model (i.e. number of shared parameters).

        """
        # collects num. of params at each layer. usefeul class attribute for later use (see self._grad2vec)
        self.grad_index = []
        # NOTE: get_share_params() is defined in AbsArchitecture
        for param in self.get_share_params():
            # append number of params in this layer
            self.grad_index.append(param.data.numel())
        # return the total number of shared parameters
        self.grad_dim = sum(self.grad_index)
        # record this calculation was done once
        self._calc_shared_param = True

    def _grad2vec(self):
        """Obtain accumulated gradients in shared parameters of the model and return 
            in a vector form.

        Returns:
        gradient of shared parameters (as a vector).

        """
        # initialize grad with a vecotr with size num. shared param.
        grad = torch.zeros(self.grad_dim)
        # count params to put grad blocks in correct index of vector
        count = 0
        # collect grads layerwise in shared param (e.g. encoder), then put it into vector
        # NOTE: get_share_params() is defined in AbsArchitecture
        for param in self.get_share_params():
            # collect grad only if not None, else return zero vector
            if param.grad is not None:
                # calculate vecotr block start and end indices
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                # put flattened grad param into the vector block
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        # if no seperate representation gradient (current case of interest)
        if not rep_grad:
            # initilize gradient container of dim: task_num x number of shared parameters (e.g. encoder parameters)
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            # for each taks loss, calc gradient w.r.t. shared param
            for tn in range(self.task_num): 
                if mode == 'backward':
                    # accumulate gradients at both shared AND task specific parameters
                    losses[tn].backward(retain_graph=True) # TODO: commented the following part to facilitate MoDo, need to handle more efficiently: if (tn+1)!=self.task_num else losses[tn].backward()
                    # collect the shared param gradient, which is the only part conflicting
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    # calculate gradient w.r.t. shared parameter
                    # ? how none shared part (if any) is updated? 
                    # ?   -maybe this mode cannot be used for such cases? or 
                    # ?   -task specific gradient is calculated some place else?
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    # return a vecotrized gradient for each task
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                # clears the gradient of the current task, making room to accumulate gradient for next task
                self.zero_grad_share_params()
        # not used (at least for now)
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        # not used (at least now)
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            # compute num. of shared parameters (total and layer-wise)
            # ! may not need to do this at each grad computation, since the model is fixed
            # TODO see whether limiting this improve code performance
            if not self._calc_shared_param:
                self._compute_grad_dim()
            # collect grad of shared params
            # task specific param gradient is accumulated at resp. params, at least with 'backward' mode
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            # update the shared parameters
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass
