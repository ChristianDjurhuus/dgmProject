import os
import logging
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from trainer import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from data import MNISTDataModule

BATCH_SIZE = 2
MAX_EPOCHS = 10
idx = 1
emb_dim = 32
hidden_dim = 128


logging.getLogger("lightning").setLevel(logging.WARNING)
if not os.path.exists("/runs"):
    os.makedirs("/runs")
if not os.path.isdir("/runs" + "/run{}/".format(idx)):
    print("Creating directory")
    os.mkdir("/runs" + "/run{}/".format(idx))
print("Starting Run {}".format(idx))
checkpoint_callback = ModelCheckpoint(
    dirpath="/runs" + "/run{}/".format(idx),
    save_top_k=1,
    monitor="val_loss",
    save_last=True,
)
lr_logger = LearningRateMonitor()
tb_logger = TensorBoardLogger("/runs" + "/run{}/".format(idx))


model = Trainer(emb_dim = emb_dim, hidden_dim = hidden_dim)

datamodule = MNISTDataModule(
    batch_size=BATCH_SIZE,
    data_dir = "data"
)

tqdm_progress_bar = TQDMProgressBar(refresh_rate=20)
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    callbacks=[tqdm_progress_bar, lr_logger, checkpoint_callback],
)
trainer.fit(model=model, datamodule=datamodule)
