import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import Adam
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import Compose, Normalize, ToTensor  # type: ignore

from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.module import AutoEncoderPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        # encode
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        # decode
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x


def get_data():
    # Fix: https://stackoverflow.com/a/66820249
    MNIST.resources = [
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(download=True, root=".", transform=data_transform, train=True)

    val = MNIST(download=False, root=".", transform=data_transform, train=False)

    return train, val


if __name__ == "__main__":

    EXPERIMENT_NAME = "mnist-autoencoder"

    configure_logging(f"logs/{EXPERIMENT_NAME}")

    train, test = get_data()

    ldm = PLDataModuleFromDatasets(
        train, test=test, batch_size=128, batch_size_eval=256
    )

    model = Net()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    lm = AutoEncoderPLModule(model, optimizer, criterion)

    trainer = make_trainer(
        EXPERIMENT_NAME, max_epochs=5, gpus=1, wandb_project="testpl"
    )
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
