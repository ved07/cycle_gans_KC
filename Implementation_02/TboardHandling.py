from torch.utils.tensorboard import SummaryWriter
import torchvision


def ImageDisplay(DataLoader, directory, label=""):
    writer = SummaryWriter("runs/"+directory)
    example = iter(DataLoader)
    example = example.next()[0]
    example = torchvision.utils.make_grid(example)
    writer.add_image(label, example)
    writer.close()
