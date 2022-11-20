
import torch
import numpy as np

from tqdm import trange
from collections import defaultdict

from vae import VariationalAutoEncoder, VariationalInference
from src.mnist_loader import load_mnist


from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def train(data: dict, vae: torch.nn.Module, vi: torch.nn.Module,
          optimizer: torch.optim.Optimizer, epochs=500, device=torch.device('cpu'),
          checkpoint_every=50, val_every_epoch=50,
          experiment_name='', tensorboard_logdir='../logs'):

    # Setup tensorboard
    writer = SummaryWriter(Path(f"{tensorboard_logdir}")/experiment_name)

    # Initialize performance storages
    train_performances = defaultdict(list)
    val_performances = defaultdict(list)

    # Run through epochs
    with trange(epochs) as t:
        for epoch in t:

            # Initialize training for epoch
            vae.train()
            train_performances_epoch = defaultdict(list)
            loss_epoch = []
            for batch_id in data['train'].keys():
                # Define data
                x = data['train'][batch_id]['data'].to(device)
                y = data['train'][batch_id]['labels'].to(device)

                # Compute loss
                loss, diagnostics, outputs = vi(vae, x)
                loss_epoch.append(loss.item())

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Gather information from batch
                for k, v in diagnostics.items():
                    train_performances_epoch[k] += [v.mean().item()]

            # Summarize epoch performance
            for k, v in train_performances_epoch.items():
                train_performances[k] += [np.mean(train_performances_epoch[k])]

            #if epoch % store_train_every_epoch == 0:
            # Add information to tensorboard
            writer.add_scalar('TRAIN/Loss', np.mean(loss_epoch), epoch)
            writer.add_scalar('TRAIN/elbo', np.mean(train_performances_epoch['elbo']), epoch)
            writer.add_scalar('TRAIN/log_px', np.mean(train_performances_epoch['log_px']), epoch)
            writer.add_scalar('TRAIN/kl', np.mean(train_performances_epoch['kl']), epoch)

            if epoch % checkpoint_every == 0:
                # Store checkpointed model
                vae.to(torch.device('cpu'))
                torch.save(VAE, f"models/{experiment_name}/{epoch}.ckpt")
                vae.to(device)

            if epoch % val_every_epoch == 0:
                # Evaluate on a single test batch
                with torch.no_grad():
                    vae.eval()

                    # Randomly select validation batch
                    val_batch_id = np.random.choice(list(data['test'].keys()))

                    # Define data
                    x = data['test'][val_batch_id]['data'].to(device)
                    y = data['test'][val_batch_id]['labels'].to(device)

                    # Compute loss
                    loss, diagnostics, outputs = vi(vae, x)
                    # Add information to tensorboard
                    writer.add_scalar('VALIDATION/Loss', loss, epoch)

                    # Gather validation epoch
                    for k, v in diagnostics.items():
                        val_performances[k] += [v.mean().item()]
                        # Add information to tensorboard
                        writer.add_scalar(f'VALIDATION/{k}', v.mean(), epoch)

            # Print status
            t.set_description_str(f'Training ELBO: {train_performances["elbo"][-1]:.3f} \t| \t Validation ELBO: {val_performances["elbo"][-1]:.3f} | Progress')

    # Close tensorboard
    writer.close()

if __name__ == '__main__':

    import os
    os.makedirs("models", exist_ok=True)

    # Define name for saving model and performance
    experiment_name = input("Enter experiment name: ")

    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load MNIST
    mnist = load_mnist(data_path="../data", batch_size=64)

    # Define VAE and inference object
    VAE = VariationalAutoEncoder(observation_shape=(28, 28), z_dim=32).to(device)
    VI = VariationalInference(beta=1)

    # Setup optimizer
    optimizer = torch.optim.Adam(VAE.parameters(), lr=1e-3)

    # Run training
    train(mnist, VAE, VI, optimizer, epochs=100, device=device,
          val_every_epoch=5, checkpoint_every=20,
          tensorboard_logdir='../logs', experiment_name=experiment_name)

    # Save model
    VAE.to(torch.device('cpu'))
    torch.save(VAE, f"models/{experiment_name}_final.pt")