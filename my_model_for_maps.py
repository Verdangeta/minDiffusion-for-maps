from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm
import os

from PIL import Image

import zipfile
import gdown

import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

# Функция скачивающая и разархивирующая датасет
def downloader_maps()-> None:
    newpath = "./data"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    url = 'https://drive.google.com/uc?id=1dyEskBDpIoaf7uJFk6mUV23eOF2mVid_'
    output = './data/my_dataset.zip'
    gdown.download(url, output, quiet=False)
    
    with zipfile.ZipFile('./data/my_dataset.zip', 'r') as zip_ref:
        zip_ref.extractall("./data")
    



# My dataset class for my maps dataset which inherits torch.utils.data.Dataset

class MapsDataset(Dataset):
    """ Highway maps dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.im_names = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.im_names[idx])
        image =torchvision.io.read_image(img_name)/255

        if self.transform:
            image = self.transform(image)

        return image



def train_maps(
   path_to_dir, n_epoch: int = 100, device: str = "cuda:0",
   load_pth: Optional[str] = None , Flip: bool = False,
   lr: float = 5e-5
) -> None:
    """
    Args:
        path_to_dir (string): Directory with all the images.
        load_pth (string): Path to file with dicts of weights.
        Flip (bool): If true, uses random vertical and horizontal flips with probability equal to 0.5.
        lr (float): The value of the learning rate.
    """

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth)) # В исходном коде тут была ошибка небольшая, не использовалася load_pth

    ddpm.to(device)
    #Добавил возможность использовать Flip по желанию
    if Flip:
        tf = transforms.Compose(
            [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.RandomVerticalFlip(),
             transforms.RandomHorizontalFlip()
            ])
    else:
         tf = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    maps_dataset = MapsDataset(root_dir = path_to_dir,
                   transform = tf
                   )
    
    # Настроил batch_size так, чтобы он был максимальным и не переполнялась память GPU (чтобы вообще работал код =) )
    dataloader = DataLoader(maps_dataset, batch_size=128,
                            shuffle=True, num_workers=2)
    
    # Добавил адаптивный lr, чтобы была возможность его менять, 
    # ведь обучение не очень стабильно (это же написано в оригинальной статье)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.96 * loss_ema + 0.04 * loss.item()  
                # Увеличил  decay factor с 0.9 до 0.96, так как в оригинальной статье использовался 0.99 
                # (пробовал менять, особо не влияет на обучение)
                
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            # Сделал так, чтобы картинки генерировались не после каждой эпохи, так как процесс генерации занимает много времени
            if (i+1)%10 ==0:
                xh = ddpm.sample(8, (3, 64, 64), device) # Изменил размер генерации под текущую выборку
                xset = torch.cat([xh, x[:8]], dim=0)
                
                grid = make_grid(xset, normalize=True, nrow=4, scale_each=True) 
                # Сделал так, чтобы каждая картинка нормализовалась отдельно, в ином случае появлялись артефакты
                
                save_image(grid, f"ddpm_sample_maps_10000_{i}.png")

            # save model
 
            torch.save(ddpm.state_dict(),"./data/ddpm_maps_10000.pth")



if __name__ == "__main__":
    
    # Скачивание датасета
    downloader_maps() 
    
    
    # Первый этап обучения с lr = 5e-5
    path_to_dir = "./data/my_dataset"
    train_maps(path_to_dir,Flip = True,n_epoch = 100)

    # Второй этап обучения с lr = 1e-5
    load_pth = "./data/ddpm_maps_10000.pth"
    train_maps(path_to_dir,Flip = True,load_pth = load_pth,n_epoch = 10,lr = 1e-5)




    # Рисование картинок
    device = "cuda:0"

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    ddpm.load_state_dict(torch.load("./data/ddpm_maps_10000.pth"))

    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        xh = ddpm.sample(25, (3, 64, 64), device)
        grid = make_grid(xh, normalize=True, nrow=5,scale_each= True)
        save_image(grid, f"ddpm_sample_maps_1000_fitted.png")

