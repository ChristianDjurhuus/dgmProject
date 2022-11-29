import glob, os
from PIL import Image

from pprint import pprint
from collections import Counter

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE

from mnist_loader import MNISTDataset, get_loaders
from vanilla_vae.vae import VariationalAutoEncoder


def show_reconstructions(dataloaders: dict, dataset_type: str, model: torch.nn.Module, digit=None, device: str='cpu', nrows: int=5, ncols: int=2, figsize: tuple=(12, 15), seed: int=42):
    np.random.seed(seed)

    # Get data loader
    data = dataloaders[dataset_type]
    if digit != None:
        cond = data.dataset.labels == digit
        data.dataset.labels = data.dataset.labels[cond]
        data.dataset.dataset = data.dataset.dataset[cond]

    # Create figure
    fig, axs = plt.subplots(nrows, 2 * ncols, figsize=(figsize))

    # Plot reconstructions
    for i in range(nrows):
        for j in range(ncols):
            # Get image and label

            idx = np.random.choice(data.dataset.__len__())
            sample = data.dataset.__getitem__(idx)

            # Original and reconstruction
            x_ = sample['data'].unsqueeze(0).to(torch.device(device))
            x_hat = model(x_)['px'].mean.detach().view(28, 28)

            axs[i, 2 * j].imshow(x_.cpu().view(28,28), cmap='gray')
            axs[i, 2 * j].set_title(f"Original image ({int(sample['label'])})")
            axs[i, 2 * j].axis('off')

            axs[i, 2 * j + 1].imshow(x_hat.cpu(), cmap='gray')
            axs[i, 2 * j + 1].set_title(f"Reconstructed image ({int(sample['label'])})")
            axs[i, 2 * j + 1].axis('off')
    return fig

def show_tsne_latent_space(dataloaders, model, dataset_type, n_iter=500, N=12000, figsize=(10, 6)):
    # Get labels
    data = dataloaders[dataset_type]
    labels = data.dataset.labels[:N].numpy()
    print("\nDISTRIBUTION OF LABELS:")
    pprint(Counter(labels))

    # Get model output
    print("\nOBTAINING LATENT REPRESENTATIONS...")
    output = model(data.dataset.dataset[:N])
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

def plot_reconstructed_digits(dataloaders: dict, dataset_type: str, model: torch.nn.Module, epoch: int, N=10, figsize=(10, 10), device='cpu', seed=42):
    np.random.seed(seed)

    # Get data loader
    data = dataloaders[dataset_type]

    # Create figure
    nrows, ncols = 10, N
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize), sharex=True, sharey=True, dpi=200)
    if epoch == 0:
        true_fig, true_axs = plt.subplots(nrows, ncols, figsize=(figsize), sharex=True, sharey=True, dpi=200)

    for digit in range(0, 10):
        dataset_ = data.dataset[data.dataset.labels == digit]

        # Plot reconstructions
        for j in range(ncols):
            # Get image and label
            idx = np.random.choice(dataset_['data'].__len__())

            # Original and reconstruction
            x_ = dataset_['data'][idx].unsqueeze(0).to(torch.device(device))
            x_hat = model(x_)['px'].mean.detach().view(28, 28)

            axs[digit, j].imshow(x_hat.cpu(), cmap='gray')
            axs[digit, j].axis('off')
            if j == 0:
                axs[digit, j].set_ylabel(f"DIGIT = {int(dataset_['label'][idx].item())})", rotation=0)

            if epoch == 0:
                true_axs[digit, j].imshow(x_.cpu().view(28, 28), cmap='gray')
                true_axs[digit, j].axis('off')
    if epoch == 0:
        return fig, true_fig
    else:
        return fig, None


# Found here: https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/
def make_gif(img_dir, filename, duration=100):
    frames = [Image.open(image) for image in glob.glob(f"{img_dir}/*.png") if 'GROUND_TRUTH' not in image]
    epoch_frame = [int(("").join([char for char in image if char.isdigit()][4:])) for image in sorted(glob.glob(f"{img_dir}/*.png")) if 'GROUND_TRUTH' not in image]
    frames = list(list(zip(*sorted(zip(epoch_frame, frames), key=lambda x: x[0])))[1])
    frame_one = frames[0]
    for _ in range(20):
        frames.insert(0, frame_one)
    cwd = os.getcwd()
    os.chdir(img_dir)
    frame_one.save(f"../{filename}", format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)
    print(f"Created GIF: {filename}")
    os.chdir(cwd)

if __name__ == '__main__':

    # Specify experiment name
    experiment_name = input("Enter experiment name: ")

    # Load original mnist
    mnist_loaders = get_loaders(MNISTDataset, data_path="data", version='original')
    # Load mnist (rotated)
    mnist_rot_loaders = get_loaders(MNISTDataset, data_path="data", version='rotated')
    
    # Load model
    filename = f"vanilla_vae/models/{experiment_name}/best.ckpt"
    model = VariationalAutoEncoder()
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    model.eval()

    for name_, loaders in {'regular': mnist_loaders, 'rotated': mnist_rot_loaders}.items():
        # Get reconstructions on test set
        fig, gt_fig = plot_reconstructed_digits(loaders, 'test', model, N=10, epoch=0) # epoch=0 for getting ground truth image
        
        # Ground truth images
        gt_fig.tight_layout(rect=[0, 0, 1, 0.95])
        gt_fig.suptitle(f"{name_} - ground truth")
        gt_fig.savefig(f"plots/ground_truth_{name_}.png")
        gt_fig.show()

        # Reconstructed images
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(f"{name_} - reconstructions")
        fig.savefig(f"plots/recon_{name_}.png")
        fig.show()
        

        # t-SNE 
        fig = show_tsne_latent_space(loaders, model, dataset_type='test', N=5000, n_iter=1000)
        fig.suptitle(f"t-SNE - {name_} MNIST")
        plt.savefig(f"plots/tSNE_{name_}.png")
        fig.show()
        

    # Create a gif og validation reconstructions from training
    make_gif(f"plots/vanilla_vae/{experiment_name}/val_reconstructions", filename="val_reconstructions.gif")



    # # Check model performance at various epochs
    # for ckpt_num in ['0', '5', '10', '15', 'best']:
    #     file_type = 'ckpt'
    #     if ckpt_num == 'final':
    #         file_type = 'pth'
    #     filename = f"vanilla_vae/models/{experiment_name}/{ckpt_num}.{file_type}"
    #
    #     # Load model
    #     model = VariationalAutoEncoder()
    #     state_dict = torch.load(filename)
    #     model.load_state_dict(state_dict)
    #     model.eval()
    #
    #     # Show reconstructions
    #     fig = show_reconstructions(mnist_loaders, model, dataset_type='test', figsize=(8, 10))
    #     fig.suptitle(f"CHECKPOINT at {ckpt_num} EPOCH(s)\n", fontsize=15)
    #     plt.tight_layout()
    #     plt.show()
    #
