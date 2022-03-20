# import dependencies
from pathlib import Path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

path = Path().parent.absolute().__str__()
path = path.split("\\")[:-1]
path.append("Data")
path = "".join([item+"/" for item in path])


class Compose:
    def __init__(self, transformations):
        self.transforms = transformations

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image), target
        return image, 1


composed_transforms = Compose([transforms.ToTensor(), transforms.RandomCrop((256, 1024))])
Kitti_Dataset = datasets.Kitti(root=path, transforms=composed_transforms)


def Kitti_DataLoader(batch_size):
    return DataLoader(Kitti_Dataset, batch_size=batch_size, shuffle=True)

