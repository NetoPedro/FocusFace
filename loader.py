from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as TF

def wif(id):
        #np.random.seed((id + torch.initial_seed()) % np.iinfo(np.int32).max)
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

class FaceDatasetVal(ImageFolder):

    def __init__(self, root, transform=None, loader=datasets.folder.default_loader, is_valid_file=None,prob = 1.0):
        super(FaceDatasetVal, self).__init__(root, transform=transform,is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.prob = prob
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        original_path = path
        rand = np.random.uniform()
        add_mask = False
        if rand < self.prob:
            add_mask = True
            path = path.replace("imgs","imgs_masked2")
        else:
            add_mask = False
            
        try: 
            sample = self.loader(path)
        except:
            try:
                sample = self.loader(path.replace(".jpg","_surgical.jpg"))
            except:
                try:
                    sample = self.loader(path.replace(".jpg","_cloth.jpg"))    
                except:
                    try:
                        sample = self.loader(path.replace(".jpg","_N95.jpg"))
                    except:
                        try:
                            sample = self.loader(path.replace(".jpg","_KN95.jpg"))
                        except:
                            add_mask = False
                            sample = self.loader(original_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        mask = 0
        if add_mask:
            mask = 1
        sample = {'image': sample, 'identity': target,'mask':mask}
        return sample

class FaceDataset(ImageFolder):

    def __init__(self, root, transform=None, loader=datasets.folder.default_loader, is_valid_file=None,prob = 1.0):
        super(FaceDataset, self).__init__(root, transform=transform,is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.prob = prob
        self.transforms2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        original_path = path

        add_mask = False
        path = path.replace("imgs","imgs_masked2")
        
        try: 
            sample = self.loader(path)
        except:
            try:
                sample = self.loader(path.replace(".jpg","_surgical.jpg"))
            except:
                try:
                    sample = self.loader(path.replace(".jpg","_cloth.jpg"))    
                except:
                    try:
                        sample = self.loader(path.replace(".jpg","_N95.jpg"))
                    except:
                        try:
                            sample = self.loader(path.replace(".jpg","_KN95.jpg"))
                        except:
                            add_mask = True
                            sample = self.loader(original_path)

        unmasked_sample = self.loader(original_path)

        if self.transform is not None:
            sample = self.transform(sample)
            unmasked_sample = self.transform(unmasked_sample)
            if np.random.uniform() > 0.5:
                sample = TF.hflip(sample)
                unmasked_sample = TF.hflip(unmasked_sample)
            sample = self.transforms2(sample)
            unmasked_sample = self.transforms2(unmasked_sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        mask = 1
        if add_mask:
            mask = 0
        
        sample = {'image_masked': sample, 'identity': target,'mask':mask,'image':unmasked_sample}
        return sample

#"python mask_the_face.py --path ../Masked-Face-Recognition2/faces_emore_mask/ --code cloth, surgical-#adff2f, surgical-#87cefa, KN95, N95"
#"surgical green, surgical blue, N95, cloth, and KN95"

def get_train_dataset(imgs_folder):
    train_transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.CenterCrop((112,112))
    ])
    ds = FaceDataset(imgs_folder, train_transform,prob=0.55)
    class_num = ds[-1]["identity"] + 1
    return ds, class_num

def get_valid_dataset(imgs_folder):
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = FaceDatasetVal(imgs_folder, valid_transform,prob=0)
    class_num = ds[-1]["identity"] + 1
    return ds, class_num


def get_train_loader(batch_size,workers,validation_split):
    
    ds, class_num = get_train_dataset("/home/pcarneiro/Masked-Face-Recognition2/faces_emore/imgs")
    ds_val, class_num = get_valid_dataset("/home/pcarneiro/Masked-Face-Recognition2/faces_emore/imgs")
    shuffle_dataset = True
    np.random.seed(25)
    dataset_size = len(ds)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.shuffle(indices)
    
    _, val_indices = indices[split:], indices[:split]
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(ds, batch_size=batch_size, 
                                        shuffle=True,num_workers=workers,pin_memory=True,worker_init_fn=wif)

    validation_loader = DataLoader(ds_val, batch_size=batch_size,
                                        sampler=valid_sampler,num_workers=workers,shuffle=False,pin_memory=True,worker_init_fn=wif)
    return train_loader,validation_loader, class_num 



