import torch
import torch.nn as nn

def model_input_test():
    # Define a simple neural network module
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 3)
            self.fc3 = nn.Linear(3, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Instantiate the module
    my_module = MyModule()

    # Get shared parameters
    shared_params = []
    for name, param in my_module.named_parameters():
        # if '.' in name:  # Filter out parameters of child modules
        #     continue
        if param.requires_grad:
            shared_params.append([name, param])
    # Print shared parameters
    for name, param in shared_params:
        print(f'\n{name}')
        print(param)

    # Testing dummy module
    dummy_input = torch.rand(size=[10,])
    print('dummy input to model :', dummy_input)
    print('model output to dummy input :', my_module(dummy_input))

def list_to_torch_tensor_test():
    cude_device1 = 1
    cude_device2 = 1
    device1 = torch.device(f'cuda:{cude_device1}') if torch.cuda.is_available() else 'cpu'
    device2 = torch.device(f'cuda:{cude_device2}') if torch.cuda.is_available() else 'cpu'
    l1 = torch.rand([10,]).to(device=device1)
    l2 = torch.rand([10,]).to(device=device2)

    print(f'\ncuda{cude_device1} torch tensor 1, l1:', l1)
    print(f'\ncuda{cude_device2} torch tensor 2, l2:', l2)

    # append tensors to a list
    l = []
    l.append(l1)
    l.append(l2)

    print('\nlist containing l1 and l2, l:', l)

    l_tensor = torch.stack(l)

    print('\nl after torch.stack, l_tensor:', l_tensor)
    print('\ndevice of l_tensor:', l_tensor.device)

    print('\nelemwntwise sum over l1 and l2:', torch.sum(l_tensor, dim=0))
    print('\naccess 0 index (row) of l_tensor:', l_tensor[0])

    print('\nConcludions:\
        \n1. When torch.stack is used, the tensor created will', \
        'automatically be assigned to the device of the elemnt tensors, if all the element tensors are from same device \
        \n2. If two tensors are from different devices, pytorch will throw an error: \
        \nRuntimeError: All input tensors must be on the same device. Received cuda:0 and cuda:1')

list_to_torch_tensor_test()   