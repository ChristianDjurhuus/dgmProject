import glob, os
from PIL import Image

from pprint import pprint
from collections import Counter, defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA


from mnist_loader import MNISTDataset, get_loaders
from vanilla_vae.vae import VariationalAutoEncoder
from invariant_vae.invariant_vaeV2 import VAE


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

def pca_tsne(dataloaders, model, dataset_type, n_iter=500, N=12000, figsize=(10, 6)):
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
    print("\nRUNNING PCA...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data.iloc[:, :32])
    data['pca1'] = pca_results[:, 0]
    data['pca2'] = pca_results[:, 1]
    print("\nPCA SUCCESSFULLY FINISHED!")

    # Run t-SNE
    print("\nRUNNING T-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=n_iter)
    tsne_results = tsne.fit_transform(data.iloc[:, :32])
    data['tsne1'] = tsne_results[:, 0]
    data['tsne2'] = tsne_results[:, 1]
    print("\nT-SNE SUCCESSFULLY FINISHED!")

    plt.style.use('ggplot')
    fig_pca = plt.figure(figsize=figsize)
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.7)

    fig_tsne = plt.figure(figsize=figsize)
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.7)

    return fig_pca, fig_tsne


def latent_space_analysis(dataloaders, model, dataset_type, K_NEIGHBOURS=np.linspace(1, 100, 11, dtype=int), n_iter=500, N=12000, figsize=(10, 6)):
    plt.style.use('ggplot')

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
    print("\nRUNNING PCA...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data.iloc[:, :32])
    data['pca1'] = pca_results[:, 0]
    data['pca2'] = pca_results[:, 1]
    print("\nPCA SUCCESSFULLY FINISHED!")

    # Run t-SNE
    print("\nRUNNING T-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=n_iter)
    tsne_results = tsne.fit_transform(data.iloc[:, :32])
    data['tsne1'] = tsne_results[:, 0]
    data['tsne2'] = tsne_results[:, 1]
    print("\nT-SNE SUCCESSFULLY FINISHED!")

    fig_pca = plt.figure(figsize=figsize)
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.7)

    fig_tsne = plt.figure(figsize=figsize)
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.7)


    from sklearn.model_selection import KFold
    from sklearn.metrics import balanced_accuracy_score


    print("\nKNN - PCA")
    predictions = defaultdict(dict)
    predictions['labels'] = data['labels'].values
    results = defaultdict(dict)
    res_pca = defaultdict(dict)
    for k_neighbors in K_NEIGHBOURS:
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        results[k_neighbors] = cross_val_score(knn, data[['pca1', 'pca2']], data['labels'], cv=5)
    # print mean and std
    results = {key: (val.mean(), val.std()) for key, val in results.items()}
    pprint(results)
    res_pca['all'] = results

    # Predict labels with KNN on PCA
    K_pca = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[0][0]
    knn = KNeighborsClassifier(n_neighbors=K_pca)
    knn.fit(data[['pca1', 'pca2']], data['labels'])
    predictions['pca'] = knn.predict(data[['pca1', 'pca2']])

    print("\nKNN - t-SNE")
    results = defaultdict(dict)
    res_tsne = defaultdict(dict)
    for k_neighbors in K_NEIGHBOURS:
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        results[k_neighbors] = cross_val_score(knn, data[['tsne1', 'tsne2']], data['labels'], cv=5)
    # print mean and std
    results = {key: (val.mean(), val.std()) for key, val in results.items()}
    pprint(results)
    res_tsne['all'] = results

    # Predict labels with KNN on PCA
    K_tsne = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[0][0]
    knn = KNeighborsClassifier(n_neighbors=K_tsne)
    knn.fit(data[['tsne1', 'tsne2']], data['labels'])
    predictions['tsne'] = knn.predict(data[['tsne1', 'tsne2']])


    print("\nKNN - 32 latent dim.")
    results = defaultdict(dict)
    res_latent = defaultdict(dict)
    for k_neighbors in K_NEIGHBOURS:
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        results[k_neighbors] = cross_val_score(knn, data.iloc[:, :32], data['labels'], cv=5)
    # print mean and std
    results = {key: (val.mean(), val.std()) for key, val in results.items()}
    pprint(results)
    res_latent['all'] = results

    # Predict labels with KNN on PCA
    K_latent = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[0][0]
    knn = KNeighborsClassifier(n_neighbors=K_latent)
    knn.fit(data.iloc[:, :32], data['labels'])
    predictions['full_latent'] = knn.predict(data.iloc[:, :32])

    res_pca = defaultdict(dict)
    res_tsne = defaultdict(dict)
    res_latent = defaultdict(dict)
    pca_acc, tsne_acc, latent_acc = [], [], []
    for train_idx, test_idx in KFold(n_splits=5).split(pd.DataFrame(predictions)):

        pca_acc.append(balanced_accuracy_score(data.iloc[test_idx]['labels'], predictions.iloc[test_idx]['pca']))
        tsne_acc.append(balanced_accuracy_score(data.iloc[test_idx]['labels'], predictions.iloc[test_idx]['tsne']))
        latent_acc.append(balanced_accuracy_score(data.iloc[test_idx]['labels'], predictions.iloc[test_idx]['full_latent']))

    res_pca[f'total ({K_pca} neighbors)'] = (np.mean(pca_acc), np.std(pca_acc) / np.sqrt(5))
    res_tsne[f'total ({K_tsne} neighbors)'] = (np.mean(tsne_acc), np.std(tsne_acc) / np.sqrt(5))
    res_latent[f'total ({K_latent} neighbors)'] = (np.mean(latent_acc), np.std(latent_acc) / np.sqrt(5))

    predictions = pd.DataFrame(predictions)
    for digit in range(0, 10):
        predictions_ = predictions[predictions['labels'] == digit]

        pca_acc = []
        tsne_acc = []
        latent_acc = []
        for train_idx, test_idx in KFold(n_splits=5).split(pd.DataFrame(predictions_)):

            pca_acc.append(balanced_accuracy_score(predictions_.iloc[test_idx]['labels'], predictions_.iloc[test_idx]['pca']))
            tsne_acc.append(balanced_accuracy_score(predictions_.iloc[test_idx]['labels'], predictions_.iloc[test_idx]['tsne']))
            latent_acc.append(balanced_accuracy_score(predictions_.iloc[test_idx]['labels'], predictions_.iloc[test_idx]['full_latent']))

        res_pca[f'digit{digit}'] = (np.mean(pca_acc), np.std(pca_acc) / np.sqrt(5))
        res_tsne[f'digit{digit}'] = (np.mean(tsne_acc), np.std(tsne_acc) / np.sqrt(5))
        res_latent[f'digit{digit}'] = (np.mean(latent_acc), np.std(latent_acc) / np.sqrt(5))

    return fig_pca, fig_tsne, pd.DataFrame(res_pca, index=['mean bal. acc.', 'sem']).T, pd.DataFrame(res_tsne, index=['mean bal. acc.', 'sem']).T, pd.DataFrame(res_latent, index=['mean bal. acc.', 'sem']).T


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
            x_hat = model(x_)['px'].mean.detach().view(28, 28) if isinstance(model, VariationalAutoEncoder) else model(x_)['px'].detach().view(28, 28)

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
    model_type = input("Choose model type, 1 for vanilla, 2 for invariant: ")
    model_name = "vanilla_vae" if model_type == "1" else "invariant_vae"
    experiment_name = input("Enter experiment name: ")
    N = 5000

    enc_hidden_dim = 128 if "128" in experiment_name else 32
    dec_hidden_dim = 128

    # Load original mnist
    mnist_loaders = get_loaders(MNISTDataset, data_path="data", version='original')
    # Load mnist (rotated)
    mnist_rot_loaders = get_loaders(MNISTDataset, data_path="data", version='rotated')

    # Load model
    filename = f"{model_name}/models/{experiment_name}/best.ckpt"
    model = VariationalAutoEncoder(enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim) if model_type == "1" else VAE(32, 32)
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict)
    model.eval()

    # for name_, loaders in {'regular': mnist_loaders, 'rotated': mnist_rot_loaders}.items():
    #
    #     # Run analysis of latent space
    #     fig_pca, fig_tsne = pca_tsne(loaders, model, dataset_type='test', N=N, n_iter=1000)
    #     fig_pca.suptitle(f"PCA - {name_} MNIST")
    #     fig_tsne.suptitle(f"t-SNE - {name_} MNIST")
    #     fig_pca.savefig(f"plots/PCA_{name_}.png")
    #     fig_tsne.savefig(f"plots/PCA_{name_}.png")
    #     fig_pca.show()
    #     fig_tsne.show()

    loaders_ =  {'regular': mnist_loaders, 'rotated': mnist_rot_loaders} if model_type == '1' else {'regular_invariant': mnist_loaders, 'rotated_invariant': mnist_rot_loaders}
    for name_, loaders in loaders_.items():

        # Run analysis of latent space
        fig_pca, fig_tsne, pca, tsne, latent = latent_space_analysis(loaders, model, dataset_type='test', N=N, n_iter=1000)
        fig_pca.suptitle(f"PCA - {name_} MNIST")
        fig_tsne.suptitle(f"t-SNE - {name_} MNIST")
        fig_pca.savefig(f"plots/{model_name}/{experiment_name}/PCA_{name_}.png")
        fig_tsne.savefig(f"plots/{model_name}/{experiment_name}/PCA_{name_}.png")
        fig_pca.show()
        fig_tsne.show()

        # save the results of the KNN classifier for each digit
        pca.to_csv(f"KNN_pca_{name_}.csv")
        tsne.to_csv(f"KNN_tsne_{name_}.csv")
        latent.to_csv(f"KNN_full_latent{name_}.csv")

        # Get reconstructions on test set
        fig, gt_fig = plot_reconstructed_digits(loaders, 'test', model, N=10, epoch=0)  # epoch=0 for getting ground truth image

        # Ground truth images
        gt_fig.tight_layout(rect=[0, 0, 1, 0.95])
        gt_fig.suptitle(f"{name_} - ground truth")
        gt_fig.savefig(f"plots/{model_name}/{experiment_name}/ground_truth_{name_}.png")
        gt_fig.show()

        # Reconstructed images
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(f"{name_} - reconstructions")
        fig.savefig(f"plots/{model_name}/{experiment_name}/recon_{name_}.png")
        fig.show()

    # Create a gif og validation reconstructions from training
    make_gif(f"plots/{model_name}/{experiment_name}/val_reconstructions", filename="val_reconstructions.gif")



    # for name_, loaders in {'regular_invariant': mnist_loaders, 'rotated_invariant': mnist_rot_loaders}.items():
    #     # Get reconstructions on test set
    #     fig, gt_fig = plot_reconstructed_digits(loaders, 'test', model, N=10, epoch=0) # epoch=0 for getting ground truth image
    #
    #     # Ground truth images
    #     gt_fig.tight_layout(rect=[0, 0, 1, 0.95])
    #     gt_fig.suptitle(f"{name_} - ground truth")
    #     gt_fig.savefig(f"plots/ground_truth_{name_}.png")
    #     gt_fig.show()
    #
    #     # Reconstructed images
    #     fig.tight_layout(rect=[0, 0, 1, 0.95])
    #     fig.suptitle(f"{name_} - reconstructions")
    #     fig.savefig(f"plots/recon_{name_}.png")
    #     fig.show()


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
