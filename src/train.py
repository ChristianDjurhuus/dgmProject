import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from collections import defaultdict

from vanilla_vae.vae import VariationalAutoEncoder, VariationalInference
from mnist_loader import MNISTDataset, get_loaders
from results import show_reconstructions, plot_reconstructed_digits


def train(dataloaders: dict, vae: torch.nn.Module, vi: torch.nn.Module,
          optimizer: torch.optim.Optimizer, epochs=500, device=torch.device('cpu'),
          checkpoint_every=50, val_every_epoch=50,
          experiment_name='', tensorboard_logdir='../logs', save_path=''):

    # Setup tensorboard
    os.makedirs(f"{save_path}/models/{experiment_name}", exist_ok=True)
    writer = SummaryWriter(Path(f"{save_path}/{tensorboard_logdir}")/experiment_name)

    # Initialize performance storages
    current_best_loss = np.inf
    train_performances = defaultdict(list)
    val_performances = defaultdict(list)

    # Run through epochs
    with trange(epochs) as t:
        for epoch in t:

            # Evaluate on validation set
            if epoch % val_every_epoch == 0:
                with torch.no_grad():
                    vae.eval()

                    val_loss_epoch, val_performances_epoch = [], defaultdict(list)
                    for batch_idx, batch in enumerate(dataloaders['val']):
                        # Define data
                        x = batch['data'].to(device)
                        y = batch['label'].to(device)

                        # Compute loss
                        loss, diagnostics, outputs = vi(vae, x)

                        # Store validation performance
                        val_loss_epoch.append(loss.item())
                        for k, v in diagnostics.items():
                            val_performances_epoch[k] += [v.mean().item()]

                    # Add information to tensorboard
                    writer.add_scalar('VALIDATION/Loss', loss, epoch)
                    for k, v in val_performances_epoch.items():
                        val_performances[k] += [np.mean(val_performances_epoch[k])]

                    # Add information to tensorboard
                    writer.add_scalar('VALIDATION/Loss', np.mean(val_loss_epoch), epoch)
                    writer.add_scalar('VALIDATION/elbo', np.mean(val_performances_epoch['elbo']), epoch)
                    writer.add_scalar('VALIDATION/log_px', np.mean(val_performances_epoch['log_px']), epoch)
                    writer.add_scalar('VALIDATION/kl', np.mean(val_performances_epoch['kl']), epoch)

                    if np.mean(val_loss_epoch) < current_best_loss:
                        print(f"\nNEW BEST LOSS (epoch = {epoch}): --> updated best.ckpt ")
                        torch.save(vae.state_dict(), f"{save_path}/models/{experiment_name}/best.ckpt")
                        current_best_loss = np.mean(val_loss_epoch)

                    os.makedirs(f"{save_path}/plots/vanilla_vae/{experiment_name}/val_reconstructions", exist_ok=True)
                    reconstructions, actual_samples = plot_reconstructed_digits(dataloaders, 'val', vae, N=10,
                                                                                device=device, figsize=(10, 10),
                                                                                epoch=epoch)
                    reconstructions.suptitle(f"EPOCH = {epoch}", fontsize=20, weight='bold')
                    reconstructions.tight_layout(rect=[0, 0.0, 1, 0.98])
                    reconstructions.savefig(
                        f"{save_path}/plots/vanilla_vae/{experiment_name}/val_reconstructions/epoch{epoch}.png")

                    if epoch == 0:
                        actual_samples.suptitle(f"GROUND TRUTH IMAGES", fontsize=20, weight='bold')
                        actual_samples.tight_layout(rect=[0, 0.0, 1, 0.98])
                        actual_samples.savefig(
                            f"{save_path}/plots/vanilla_vae/{experiment_name}/val_reconstructions/0_GROUND_TRUTH.png")
                    plt.close()

            vae.train()

            # Initialize training for epoch
            train_loss_epoch, train_performances_epoch = [], defaultdict(list)
            for batch_idx, batch in enumerate(dataloaders['train']):
                # Define data
                x = batch['data'].to(device)
                y = batch['label'].to(device)

                # Compute loss
                loss, diagnostics, outputs = vi(vae, x)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Gather information from batch
                train_loss_epoch.append(loss.item())
                for k, v in diagnostics.items():
                    train_performances_epoch[k] += [v.mean().item()]

            # Summarize epoch performance
            for k, v in train_performances_epoch.items():
                train_performances[k] += [np.mean(train_performances_epoch[k])]

            # Add information to tensorboard
            writer.add_scalar('TRAIN/Loss', np.mean(train_loss_epoch), epoch)
            writer.add_scalar('TRAIN/elbo', np.mean(train_performances_epoch['elbo']), epoch)
            writer.add_scalar('TRAIN/log_px', np.mean(train_performances_epoch['log_px']), epoch)
            writer.add_scalar('TRAIN/kl', np.mean(train_performances_epoch['kl']), epoch)

            # Store checkpointed model
            if epoch % checkpoint_every == 0:
                vae.to(torch.device('cpu'))
                torch.save(vae.state_dict(), f"{save_path}/models/{experiment_name}/{epoch}.ckpt")
                vae.to(device)

            # Print status
            t.set_description_str(f'Training ELBO: {train_performances["elbo"][-1]:.3f} \t| \t Validation ELBO: {val_performances["elbo"][-1]:.3f} | Progress')

    # Close tensorboard
    writer.close()

if __name__ == '__main__':

    # Define name for saving model and performance
    #experiment_name = input("Enter experiment name: ")
    experiment_name = 'vanilla-v1'
    save_path = r"C:\Users\alber\Desktop\DTU\dgm"

    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # Load MNIST
    mnist_loaders = get_loaders(MNISTDataset, data_path="data", version='original', batch_size=64)

    # Define VAE and inference object
    VAE = VariationalAutoEncoder().to(device)
    VI = VariationalInference(beta=1)

    # Setup optimizer
    optimizer = torch.optim.Adam(VAE.parameters(), lr=1e-3)

    # Run training
    train(mnist_loaders, VAE, VI, optimizer, epochs=5000, device=device,
          val_every_epoch=10, checkpoint_every=50,
          tensorboard_logdir='logs', experiment_name=experiment_name, save_path=save_path)

    # Save model
    VAE.to(torch.device('cpu'))
    torch.save(VAE.state_dict(), f"{save_path}/models/{experiment_name}/final.pth")