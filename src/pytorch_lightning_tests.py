import torch

print(torch.cuda.device_count())  # Should print 2


from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(56 * 56, 10)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Generate random data for demonstration purposes
x = torch.randn(10000, 56, 56)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32)

model = SimpleModel()

trainer = Trainer(devices=2, accelerator="auto", strategy="ddp")  # Using 2 GPUs

trainer.fit(model, dataloader)
