import os
from torchvision.datasets import ImageFolder
import numpy as np
import torch
import h5py

class NIC_VR(ImageFolder):

    def __init__(self, root_img, root_R, transform=None, target_transform=None):
        super(NIC_VR, self).__init__(root_img, transform=transform, target_transform=target_transform)
        self.lambda_rd = torch.tensor(
            [0.1/32,0.25/32,0.5/32,1/32,2/32,3/32,4/32,6/32,8/32,12/32,16/32,24/32,32/32], dtype=torch.float32)
        R = np.loadtxt(os.path.join(root_R, 'train_R.txt'))
        R = torch.from_numpy(R)
        self.R = R.float()

        # filename = os.path.join(root_R, 'train_R.txt')
        # with open(filename, 'r') as f:
        #     for line in f:
        #         [num] = line.split()
        #         self.R.append(float(num))
        # f.close()


    def __getitem__(self, index):
        path, _ = self.imgs[int(index/13)]
        img = self.loader(path)
        lambda_rd = self.lambda_rd[index % 13:(index % 13+1)]
        r = self.R[index:index+1]

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     r = self.target_transform(r)
        #     lambda_rd = self.target_transform(lambda_rd)

        return img, lambda_rd, r

    def __len__(self):
        return len(self.R)


class path_NIC(ImageFolder):
    def __init__(self, root, name, transform=None, target_transform=None):
        super(path_NIC, self).__init__(root, transform=transform, target_transform=None)
        
        with open(name,'r') as f:
            path_list = f.readlines()
        # for line in path_list:
        #     path_list[line.index] = line.strip('\n')
        for i in range(len(path_list)):
            path_list[i] = path_list[i].strip('\n')
        self.path_list = path_list
        
    def __getitem__(self, index):
        path = self.path_list[index]
        #if index>545580 and index<547200
        sample = self.loader(path)
        #else:
            #sample=np.zeros((2,2))
        if self.transform is not None:
            sample = self.transform(sample)
        # print(path)
        return sample, path
        
    def __len__(self):
        return len(self.path_list)

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["train_img"])
        self.transform = transform

    def __getitem__(self, index):
        # print(index)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["train_img"]
        img = self.dataset[index].transpose(1,2,0)
        # print(img.shape)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        # print(self.dataset_len)
        return self.dataset_len
        # return 100


