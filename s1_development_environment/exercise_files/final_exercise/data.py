import torch
import numpy as np
import glob

class dataset:
    def __init__(self,data,target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data[idx]
        y = self.target[idx]
        
        return X,y

def mnist():
    # exchange with the corrupted mnist dataset
    train_files = glob.glob("C:/Users/Siddhi/Documents/GitHub/dtu_mlops/data/corruptmnist/train*.npz")
    test_files = glob.glob("C:/Users/Siddhi/Documents/GitHub/dtu_mlops/data/corruptmnist/test*.npz")

    images = []
    labels = []
    for file in train_files:
        data = np.load(file)
        images.append(data['images'])
        labels.append(data['labels'])
    train_images = np.concatenate((images),axis=0)
    train_labels = np.concatenate((labels),axis=0)

    images = []
    labels = []
    for file in test_files:
        data = np.load(file)
        images.append(data['images'])
        labels.append(data['labels'])
    test_images = np.concatenate((images),axis=0)
    test_labels = np.concatenate((labels),axis=0)

    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).long()

    test_images = torch.from_numpy(test_images).float()
    test_labels = torch.from_numpy(test_labels).long()

    train = dataset(train_images,train_labels)
    test = dataset(test_images,test_labels) 
    return train, test
