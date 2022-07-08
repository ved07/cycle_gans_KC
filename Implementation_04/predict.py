import torch

from torchvision.utils import save_image
from DataHandling import Kitti_Dataset, Carla_Dataset
import random

device =  torch.device('cpu')
print("aaaa")
input_im = Carla_Dataset[int(input())][0].reshape([1, 3, 256, 1024]).to(device)
print("aaaa")
save_image(input_im, "in.png")
Generator = torch.load("scriptedSim2RealGen.pt").to(device)

im = Generator(input_im)
save_image(im, "2.png")