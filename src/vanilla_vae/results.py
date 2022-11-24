
from pprint import pprint
from collections import Counter

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE

from src.mnist_loader import MNISTDataset, get_loaders
from src.vanilla_vae.vae import VariationalAutoEncoder


def show_reconstructions(dataloaders, model, dataset_type: str, nrows: int=5, ncols: int=2, figsize: tuple=(12, 15), seed: int=42):
    np.random.seed(seed)

    # Get data loader
    data = dataloaders[dataset_type]

    fig, axs = plt.subplots(nrows, 2 * ncols, figsize=(figsize))
    for i in range(nrows):
        for j in range(ncols):
            # Get image and label
            idx = np.random.choice(data.dataset.__len__())
            sample = data.dataset.__getitem__(idx)

            # Original and reconstruction
            x_ = sample['data'].unsqueeze(0).view(28, 28)
            x_hat = model(sample['data'].unsqueeze(0))['px'].mean.detach().view(28, 28)

            axs[i, 2 * j].imshow(x_, cmap='gray')
            axs[i, 2 * j].set_title(f"Original image ({int(sample['label'])})")
            axs[i, 2 * j].axis('off')

            axs[i, 2 * j + 1].imshow(x_hat, cmap='gray')
            axs[i, 2 * j + 1].set_title(f"Reconstructed image ({int(sample['label'])})")
            axs[i, 2 * j + 1].axis('off')
    return fig

def show_tsne_latent_space(dataset, model, n_iter=500, N=12000, figsize=(10, 6)):
    # Get labels
    labels = dataset['train']['labels'][:N].numpy()
    print("\nDISTRIBUTION OF LABELS:")
    pprint(Counter(labels))

    # Get model output
    print("\nOBTAINING LATENT REPRESENTATIONS...")
    output = model(dataset['train']['data'][:N])
    latent_representations = output['z'].detach().numpy()
    print("\nLATENT REPRESENTATIONS OBTAINED!")

    # Gather in dataframe
    data = pd.DataFrame(latent_representations)
    data['labels'] = labels.astype(int)

    # Run t-SNE
    print("\nRUNNING T-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=n_iter)
    tsne_results = tsne.fit_transform(data)
    data['tsne1'] = tsne_results[:, 0]
    data['tsne2'] = tsne_results[:, 1]
    print("\nT-SNE SUCCESSFULLY FINISHED!")

    plt.style.use('ggplot')
    fig = plt.figure(figsize=figsize)
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.7)
    return fig

if __name__ == '__main__':

    # Specify experiment name
    experiment_name = input("Enter experiment name: ")

    # Load mnist
    mnist_loaders = get_loaders(MNISTDataset, data_path="../data", version='original')
    mnist_rot_loaders = get_loaders(MNISTDataset, data_path="../data", version='rotated')

    # Check model performance at various epochs
    for ckpt_num in ['0', '5', '10', '15', 'best']:
        file_type = 'ckpt'
        if ckpt_num == 'final':
            file_type = 'pth'
        filename = f"models/{experiment_name}/{ckpt_num}.{file_type}"

        # Load model
        model = VariationalAutoEncoder()
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict)
        model.eval()

        # Show reconstructions
        fig = show_reconstructions(mnist_loaders, model, dataset_type='test', figsize=(8, 10))
        fig.suptitle(f"CHECKPOINT at {ckpt_num} EPOCH(s)\n", fontsize=15)
        plt.tight_layout()
        plt.show()

    # Load model
    filename = f"models/{experiment_name}/85.ckpt"
    model = VariationalAutoEncoder()
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    model.eval()

    # t-SNE
    fig = show_tsne_latent_space(mnist_rot, model, N=1000)
    fig.show()


