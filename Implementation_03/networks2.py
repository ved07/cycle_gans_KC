# importing dependencies
from torch import nn
import torch
from DataHandling import Kitti_DataLoader
from torchsummary import summary


def condensing_layer(input_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(output_dim),
        nn.ReLU(inplace=True)
        )


def expanding_layer(input_dim, output_dim):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode="bilinear", align_corners=False),
        nn.Conv2d(input_dim, output_dim,kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
        nn.InstanceNorm2d(input_dim),
        nn.LeakyReLU(0.2),
    )
def residual_layer(input_dim):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(input_dim, input_dim, kernel_size=3),
        nn.InstanceNorm2d(input_dim),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(input_dim, input_dim, kernel_size=3),
        nn.InstanceNorm2d(input_dim),
        nn.ReLU(inplace=True),
    )

class Generator(nn.Module):
    def __init__(self, im_dim=3, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(im_dim),
            nn.Conv2d(im_dim, hidden_dim, kernel_size=7),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            condensing_layer(input_dim=hidden_dim,output_dim=hidden_dim*2),
            condensing_layer(input_dim=hidden_dim*2, output_dim=hidden_dim*4),
            residual_layer(input_dim=hidden_dim * 4),
            residual_layer(input_dim=hidden_dim * 4),
            residual_layer(input_dim=hidden_dim * 4),
            expanding_layer(input_dim=hidden_dim*4,output_dim=hidden_dim*2),
            expanding_layer(input_dim=hidden_dim*2,output_dim=hidden_dim),
            nn.ReflectionPad2d(im_dim),
            nn.Conv2d(hidden_dim, im_dim, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class Discriminator(nn.Module):

    def DiscBlock(self, input_dim, output_dim, normalize = True):

        if normalize:
            return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1),
                                           nn.InstanceNorm2d(output_dim),
                                           nn.LeakyReLU(0.2))
        else:
            return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1),
                                           nn.LeakyReLU(0.2))


    def __init__(self, im_dim=3, hidden_dim=4, input_shape=(256, 1024)):
        super().__init__()

        self.model = nn.Sequential(
            self.DiscBlock(im_dim, 64, normalize=False),
            self.DiscBlock(64, 128),
            self.DiscBlock(128, 256),
            self.DiscBlock(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
            nn.Sigmoid()
            )


    def forward(self, x: torch.Tensor):
        return self.model(x)

