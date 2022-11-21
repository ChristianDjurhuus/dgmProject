import os
import logging
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from trainer import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from data import MNISTDataModule

BATCH_SIZE = 32
MAX_EPOCHS = 10
idx = 1
emb_dim = 32
hidden_dim = 128



tb_logger = TensorBoardLogger("tb_logs", name = "invariaent_vae_run")


model = Trainer(emb_dim = emb_dim, hidden_dim = hidden_dim)

datamodule = MNISTDataModule(
    batch_size=BATCH_SIZE,
    data_dir = "data"
)

trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger = tb_logger,

)
trainer.fit(model=model, datamodule=datamodule)
