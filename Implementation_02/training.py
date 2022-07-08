# importing dependencies
import torch
import torch.nn as nn
import torchvision.utils

import networks2 as networks
import os
import DataHandling
import TboardHandling
import sys
device = torch.device('cuda')



print(torch.cuda.is_available())
input_shape = (256, 1024)
Sim2RealDiscriminator = networks.Discriminator(input_shape=input_shape).to(device)
Sim2RealGenerator = networks.Generator().to(device)
Real2SimDiscriminator = networks.Discriminator(input_shape=input_shape).to(device)
Real2SimGenerator = networks.Generator().to(device)
"""
Sim2RealDiscriminator = torch.load("scriptedSim2RealDisc.pt")
Real2SimDiscriminator = torch.load("scriptedReal2SimDisc.pt")
Sim2RealGenerator = torch.load("scriptedSim2RealGen.pt")
Real2SimGenerator = torch.load("scriptedReal2SimGen.pt") 
"""
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define the constants
NumEpochs = 200
lr = 0.0001
# Optimizers
Sim2RealGeneratorOptimizer = torch.optim.Adam(params=Sim2RealGenerator.parameters(), lr=lr)
Sim2RealDiscriminatorOptimizer = torch.optim.Adam(params=Sim2RealDiscriminator.parameters(), lr=lr)
Real2SimGeneratorOptimizer = torch.optim.Adam(params=Real2SimGenerator.parameters(), lr=lr)
Real2SimDiscriminatorOptimizer = torch.optim.Adam(params=Real2SimDiscriminator.parameters(), lr=lr)

KittiDataLoader = DataHandling.Kitti_DataLoader
CarlaDataLoader = DataHandling.Carla_Dataloader
advCriterion = nn.BCELoss().to(device)  # defined as the adversarial loss between the generator and discriminator
reconCriterion = nn.L1Loss().to(device)  # reconstruction of generated images loss, checking if the image is the same

cycle_lambda = 1
#identity_lambda = 1
gen_lambda = 1


def get_disc_loss(disc, fake, real, criterion):
    fakeYHat = disc(fake.detach())
    fakeLoss = criterion(fakeYHat, torch.zeros_like(fakeYHat))
    realYHat = disc(real)
    realLoss = criterion(realYHat, torch.ones_like(realYHat))
    return (realLoss + fakeLoss)/2


def get_gen_loss(disc, fake, criterion):
    discFake = disc(fake)
    loss = criterion(discFake, torch.ones_like(discFake))
    return loss


#def get_identity_loss(y, yhat, criterion):return criterion(yhat, y)



TboardHandling.ImageDisplay(KittiDataLoader, "real", "Realistic Images")
TboardHandling.ImageDisplay(CarlaDataLoader, "simulated", "Simulated Images")

for epoch in range(NumEpochs):
    for i, (KittiData, CarlaData) in enumerate(zip(KittiDataLoader, CarlaDataLoader), 0):

        SimData = CarlaData[0].to(device)
        RealData = KittiData[0].to(device)

        realFake = Sim2RealGenerator(SimData)
        simRecon = Real2SimGenerator(realFake)

        simFake = Real2SimGenerator(RealData)
        realRecon = Sim2RealGenerator(simFake)

        """
        the massive loss function, generator_loss = identity+gen+
        """
        Sim2RealDiscriminatorOptimizer.zero_grad()

        Sim2RealDiscLoss = get_disc_loss(Sim2RealDiscriminator,
                                         fake=realFake, real=RealData.detach(),
                                         criterion=advCriterion)
        Sim2RealDiscLoss.backward()
        Sim2RealDiscriminatorOptimizer.step()


        Real2SimDiscriminatorOptimizer.zero_grad()

        Real2SimDiscLoss = get_disc_loss(Real2SimDiscriminator,
                                         fake=simFake,
                                         real=SimData.detach(), criterion=advCriterion)
        Real2SimDiscLoss.backward()
        Real2SimDiscriminatorOptimizer.step()

        Sim2RealGeneratorOptimizer.zero_grad()
        Real2SimGeneratorOptimizer.zero_grad()

        SimCycleConsistencyLoss = cycle_lambda * reconCriterion(simRecon, SimData)
        RealCycleConsistencyLoss = cycle_lambda * reconCriterion(realRecon, RealData)

        # SimIdentityLoss = identity_lambda * reconCriterion(SimData, Real2SimGenerator(SimData))
        # RealIdentityLoss = identity_lambda * reconCriterion(RealData, Sim2RealGenerator(RealData))

        Sim2RealGeneratorLoss = gen_lambda * get_gen_loss(disc=Sim2RealDiscriminator,
                                                          fake=realFake, criterion=advCriterion)

        #  Real2SimGeneratorLoss = gen_lambda * get_gen_loss(disc=Real2SimDiscriminator,fake=simFake,
        #  criterion=advCriterion)

        TotalGenLoss = (Sim2RealGeneratorLoss +  # Real2SimGeneratorLoss
                        # + SimIdentityLoss + RealIdentityLoss
                        + SimCycleConsistencyLoss + RealCycleConsistencyLoss)

        TotalGenLoss.backward()

        Sim2RealGeneratorOptimizer.step()
        Real2SimGeneratorOptimizer.step()
        writer = torch.utils.tensorboard.SummaryWriter("runs/Loss")
        writer.add_scalar('Disc/Loss/train/',Real2SimDiscLoss+Sim2RealDiscLoss,i)
        writer.add_scalar('runs/Gen/Loss/train', TotalGenLoss, i)
        writer.close()
        if i % 100 == 0:
            label = f"Epoch:{epoch} Step:{i}, Gen Loss:{TotalGenLoss}" \
                    f"Disc Loss:{Real2SimDiscLoss+Sim2RealDiscLoss}"
            print(label)
            predictions = [Sim2RealGenerator(SimData[0].reshape([1, 3, input_shape[0], input_shape[1]])).reshape(
                [3, input_shape[0], input_shape[1]])]
            predictions = torchvision.utils.make_grid(predictions)
            writer = torch.utils.tensorboard.SummaryWriter("runs/" + label.replace(":", ""))
            writer.add_image(label, predictions)

            writer.close()

        else:
            print("step {} of epoch {} completed".format(i, epoch))

    scriptedSim2RealGen = torch.jit.script(Sim2RealGenerator)
    scriptedSim2RealGen.save("scriptedSim2RealGen.pt")

    scriptedSim2RealDisc = torch.jit.script(Sim2RealDiscriminator)
    scriptedSim2RealDisc.save("scriptedSim2RealDisc.pt")

    scriptedReal2SimGen = torch.jit.script(Real2SimGenerator)
    scriptedReal2SimGen.save("scriptedReal2SimGen.pt")

    scriptedReal2SimDisc = torch.jit.script(Real2SimDiscriminator)
    scriptedReal2SimDisc.save("scriptedReal2SimDisc.pt")

