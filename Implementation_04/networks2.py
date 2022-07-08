# importing dependencies
from torch import nn
import torch
from DataHandling import Kitti_DataLoader
from torchsummary import summary


def condensing_layer(input_dim, output_dim, dropout=0.25,im_shape = (256, 1024), img_shape_factor = 1):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.LayerNorm([output_dim, im_shape[0]//img_shape_factor, im_shape[1]//img_shape_factor])
        )


def expanding_layer(input_dim, output_dim, dropout=0.25, im_shape = (256, 1024), img_shape_factor = 1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode="bilinear", align_corners=False),
        nn.Conv2d(input_dim, output_dim,kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
        nn.LeakyReLU(0.2),
        nn.LayerNorm([output_dim, im_shape[0] // img_shape_factor, im_shape[1] // img_shape_factor])
    )
def residual_layer(input_dim, output_dim, dropout=0.25, img_shape_factor = 16):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.LeakyReLU(0.2),
        nn.LayerNorm([output_dim, 256 // img_shape_factor, 1024 // img_shape_factor])
    )

class Generator(nn.Module):
    def __init__(self, im_dim=3, hidden_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            condensing_layer(input_dim=im_dim,output_dim=hidden_dim, img_shape_factor=2),
            condensing_layer(input_dim=hidden_dim, output_dim=hidden_dim*2, img_shape_factor=4),
            condensing_layer(input_dim=hidden_dim*2, output_dim=hidden_dim * 4,img_shape_factor=8),
            condensing_layer(input_dim=hidden_dim*4,output_dim=hidden_dim*8, img_shape_factor=16),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            residual_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 8),
            expanding_layer(input_dim=hidden_dim*8, output_dim=hidden_dim*4, img_shape_factor=8),
            expanding_layer(input_dim=hidden_dim*4,output_dim=hidden_dim*2, img_shape_factor = 4),
            expanding_layer(input_dim=hidden_dim*2,output_dim=hidden_dim, img_shape_factor = 2),
            expanding_layer(input_dim=hidden_dim,output_dim=im_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, im_dim=3, hidden_dim=4, input_shape=(256, 1024)):
        super().__init__()

        self.model = nn.Sequential(
            condensing_layer(input_dim=im_dim, output_dim=hidden_dim, img_shape_factor=2),  # input shape halves
            condensing_layer(input_dim=hidden_dim, output_dim=hidden_dim * 2, img_shape_factor=4),  # input shape halves
            # condensing_layer(input_dim=hidden_dim * 2, output_dim=hidden_dim * 4,  img_shape_factor=8),  # input shape halves
            # condensing_layer(input_dim=hidden_dim * 4, output_dim=hidden_dim * 8),  # input shape halves
            # condensing_layer(input_dim=hidden_dim * 8, output_dim=hidden_dim * 16),  # input shape halves
            nn.Flatten(),
            nn.Linear(in_features=input_shape[0]*input_shape[1]//2, out_features=20),
            nn.Sigmoid(),
            nn.Linear(in_features=20, out_features=1),
            nn.Sigmoid()
            )


    def forward(self, x: torch.Tensor):
        return self.model(x)

