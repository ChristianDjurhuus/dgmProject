
import torch
import numpy as np

from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from vae import VariationalAutoEncoder, VariationalInference
from mnist_loader import load_mnist

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def train(data: dict, vae: torch.nn.Module, vi: torch.nn.Module,
          optimizer: torch.optim.Optimizer, epochs=500, device=torch.device('cpu'),
          store_train_every_epoch=50, val_every_epoch=50,
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
            for batch_id in data['train'].keys():
                # Define data
                x = data['train'][batch_id]['data'].to(device)
                y = data['train'][batch_id]['labels'].to(device)

                # Compute loss
                loss, diagnostics, outputs = vi(vae, x)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Add information to tensorboard
                writer.add_scalar('TRAIN/Loss', loss, epoch)

                # Gather information from batch
                for k, v in diagnostics.items():
                    train_performances_epoch[k] += [v.mean().item()]

            # Summarize epoch performance
            for k, v in train_performances_epoch.items():
                train_performances[k] += [np.mean(train_performances_epoch[k])]


            if epoch % store_train_every_epoch == 0:
                # Add information to tensorboard
                writer.add_scalar('TRAIN/elbo', np.mean(train_performances_epoch['elbo']), epoch)
                writer.add_scalar('TRAIN/log_px', np.mean(train_performances_epoch['log_px']), epoch)
                writer.add_scalar('TRAIN/kl', np.mean(train_performances_epoch['kl']), epoch)


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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # VAE and data parameters
    z_dim = 32
    batch_size = 64
    image_shape = (28, 28)

    # Load MNIST
    mnist = load_mnist(batch_size)

    # Define VAE and inference object
    VAE = VariationalAutoEncoder(image_shape, z_dim).to(device)
    VI = VariationalInference(beta=50)

    # Setup optimizer
    optimizer = torch.optim.Adam(VAE.parameters(), lr=1e-3)

    train(mnist, VAE, VI, optimizer, epochs=100, device=device,
          val_every_epoch=1, store_train_every_epoch=1,
          tensorboard_logdir='../logs', experiment_name='test_exp')


    # test
    VAE.eval()
    output = VAE(mnist['test'][10]['data'].to(device))

    import matplotlib.pyplot as plt

    nrows = ncols = 8
    fig, axs = plt.subplots(nrows, ncols)
    x_hats = output['px'].sample()
    for i in range(batch_size):
        axs[i // nrows, i % nrows].imshow(x_hats[i], cmap='gray')
    plt.show()