# importing dependencies
import torch
import torch.nn as nn
import networks
import DataHandling
import torchvision
from torch.utils.tensorboard import SummaryWriter
from networks import generate_noise
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
# defining both networks
torch.cuda.empty_cache()

Discriminator = networks.Discriminator().to(device)
Generator = networks.Generator().to(device)


loss_function = nn.BCELoss()  # Binary Cross Entropy loss

# Defining all the constants
NumEpochs = 200
lr = 0.0001
BATCH_SIZE = 8
# define both optimizers for the network, these are adam optimizers
optimizerD = torch.optim.Adam(params=Discriminator.parameters(), lr= lr)
optimizerG = torch.optim.Adam(params=Generator.parameters(), lr=lr)

# creating the dataloader object
KittiDataLoader = DataHandling.Kitti_DataLoader_Train(BATCH_SIZE)


def get_disc_loss(disc, gen, x, real, criterion):
    fake = gen.forward(x).detach()
    fakeYHat = disc.forward(fake)
    fakeLoss = criterion(fakeYHat, torch.zeros_like(fakeYHat, requires_grad=True))
    realYHat = disc.forward(real)
    realLoss = criterion(realYHat, torch.ones_like(realYHat))
    return (realLoss + fakeLoss)/2


def get_gen_loss(disc, gen, x, criterion):
    discFake = disc(gen(x))
    loss = criterion(discFake, torch.ones_like(discFake, requires_grad=True))

    return loss


for epoch in range(NumEpochs):
    for i, data in enumerate(KittiDataLoader):
        data = torch.stack([item for item in data[0]]).to(device)
        noise = generate_noise(batch_size=BATCH_SIZE).to(device)
        # zero both param gradients
        optimizerD.zero_grad()


        dLoss = get_disc_loss(Discriminator, Generator, noise, data, loss_function).to(device)

        dLoss.backward()
        optimizerD.step()

        optimizerG.zero_grad()
        gLoss = get_gen_loss(Discriminator, Generator, noise, loss_function).to(device)
        gLoss.backward()

        writer = SummaryWriter("runs/Loss/Train/")
        writer.add_scalar('Disc/', dLoss, i)
        writer.add_scalar('Gen/', gLoss, i)
        writer.close()
        optimizerG.step()
        if i % 100 == 0:
            label = f"epoch:{epoch} step:{i}"
            predictions = [Generator(noise[idx].reshape(1, 64, 16, 64)).detach().reshape(3, 256, 1024)
                           for idx in range(4)]
            predictions = torchvision.utils.make_grid(predictions)
            writer = SummaryWriter("runs/" + label.replace(":", ""))
            writer.add_image(label, predictions)
    print("epoch {} completed".format(epoch))
    scriptedGen = torch.jit.script(Generator)
    scriptedGen.save("scripted_gen.pt")
    scriptedDisc = torch.jit.script(Discriminator)
    scriptedDisc.save("scripted_disc.pt")

