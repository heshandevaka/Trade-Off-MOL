from torch.utils.data import DataLoader, Dataset
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class office_Dataset(Dataset):
    def __init__(self, dataset, root_path, task, mode, balanced=False):
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ])
        # get a more balanced image list if balanced=True
        if balanced:
            f = open('./data_txt/{}-balanced/{}_{}.txt'.format(dataset, task, mode), 'r') # reads the image list from text file
        # or the original dataset
        else:
            f = open('./data_txt/{}/{}_{}.txt'.format(dataset, task, mode), 'r') # reads the image list from text file
        self.img_list = f.readlines()
        f.close()
        self.root_path = root_path
        
    def __getitem__(self, i): # read image path and load the image  i and the label i
        img_path = self.img_list[i][:-1].split(' ')[0]
        y = int(self.img_list[i][:-1].split(' ')[1])
        img = Image.open(os.path.join(self.root_path, img_path)).convert('RGB')
        return self.transform(img), y
        
    def __len__(self):
        return len(self.img_list)
    
def office_dataloader(dataset, batchsize, root_path, balanced=False):
    # tasks if dataset is office-31 (3 tasks)
    if dataset == 'office-31':
        tasks = ['amazon', 'dslr', 'webcam']
    # tasks if dataset is office-home (4 tasks)
    elif dataset == 'office-home':
        tasks = ['Art', 'Clipart', 'Product', 'Real_World']
    # init dict of dataloasers
    data_loader = {}
    # ? this is not used, dont know the use of this
    iter_data_loader = {}
    # create and collect dataloader for each task
    for k, d in enumerate(tasks):
        # init for each task d
        data_loader[d] = {}
        iter_data_loader[d] = {}
        # create dataloaders for train, test and val datasets
        for mode in ['train', 'val', 'test']:
            shuffle = True if mode == 'train' else False
            drop_last = True if mode == 'train' else False
            # create Dataset object that load images from the path, to be used in a DataLoader object (below)
            txt_dataset = office_Dataset(dataset, root_path, d, mode, balanced=balanced)
#             print(d, mode, len(txt_dataset))
            data_loader[d][mode] = DataLoader(txt_dataset, 
                                              num_workers=0, 
                                              pin_memory=True, 
                                              batch_size=batchsize, 
                                              shuffle=shuffle,
                                              drop_last=drop_last)
            # create corresponding iter object, unclear where used
            iter_data_loader[d][mode] = iter(data_loader[d][mode])
    # return dataloaders for each task and each mode (train, test, eval)
    return data_loader, iter_data_loader
