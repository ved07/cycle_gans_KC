import torch

from torchvision.utils import save_image
from DataHandling import Kitti_Dataset, Carla_Dataset
import random

device =  torch.device('cpu')
Generator = torch.load("scriptedSim2RealGen.pt").to(device)
Generator.eval()
input_im = Carla_Dataset[int(input())][0].reshape([1, 3, 64, 256])
im = Generator.forward(input_im)
save_image(im, "2.png")
save_image(input_im, "in.png")