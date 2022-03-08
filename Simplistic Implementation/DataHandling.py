# import dependencies
from pathlib import Path
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
BATCH_SIZE = 4
path = Path().parent.absolute().__str__()
path = path.split("\\")[:-1]
path.append("Data")
path = "".join([item+"/" for item in path])

transform = [transforms.ToTensor(), transforms.CenterCrop((256, 1024))]


class Compose(object):
    def __init__(self, transform_list):
        self.transforms = transform_list

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target


transform = Compose(transform)

Kitti_Dataset = datasets.Kitti(root=path, transforms=transform)

Kitti_DataLoader = DataLoader(Kitti_Dataset, batch_size=BATCH_SIZE, shuffle=True)

