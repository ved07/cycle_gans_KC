# import dependencies
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
BATCH_SIZE = 8
path = Path().parent.absolute().__str__()
path = path.split("\\")[:-1]
carla_path = path
carla_path = "".join([item+"/" for item in carla_path])
path.append("Data")
path = "".join([item+"/" for item in path])
carla_path = carla_path+"Carla_data/"


class Compose:
    def __init__(self, transformations):
        self.transforms = transformations

    def __call__(self, image, target=1):
        for t in self.transforms:
            image, target = t(image), target
        return image, 1



composed_transforms = Compose([transforms.ToTensor(), transforms.CenterCrop((256, 1024)), transforms.Resize((64, 256))])
Kitti_Dataset = datasets.Kitti(root=path, transforms=composed_transforms)
Kitti_DataLoader = DataLoader(Kitti_Dataset, batch_size=BATCH_SIZE, shuffle=True)

class CarlaCompose:
    def __init__(self, transformations):
        self.transforms = transformations

    def __call__(self, image, target=1):
        for t in self.transforms:
            image, target = t(image), target
        return image


carla_transforms = CarlaCompose([transforms.ToTensor(), transforms.CenterCrop((256, 1024)),
                                 transforms.Resize((64, 256))])

Carla_Dataset = datasets.ImageFolder(carla_path, transform=carla_transforms)

Carla_Dataloader = DataLoader(Carla_Dataset, batch_size=BATCH_SIZE, shuffle=True)
