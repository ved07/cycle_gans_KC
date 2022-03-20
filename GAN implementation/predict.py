import torch
from networks import generate_noise
from torchvision.utils import save_image
from DataHandling import Kitti_Dataset
import random

device =  torch.device('cpu')
Generator = torch.load("scripted_gen.pt").to(device)
Generator.eval()
nois = generate_noise(1).to(device)
im = Generator.forward(nois)
save_image(im, "image{}.png".format(random.randint(1, 100)))