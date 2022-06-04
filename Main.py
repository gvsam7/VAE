"""
Author: Georgios Voulgaris
Date: 09/03/2021
Description: Test platform to validate generative models using 4 datasets (MNIST, FashionMNIST, CIFAR10, and STL10).
            The first model is a variational autoencoder comprised of convolutional and fully connected layers.
            The purpose of this code is to become a testing platform where various generative models will be tested in
            the 4 datasets.
            The datasets were chosen based on their complexity.
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
from torch.utils.data import DataLoader
import argparse
from models.VAE import VAE
from torchsummary import summary
from sklearn import decomposition, manifold
import wandb


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--latent_dims", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=3e-04)
    parser.add_argument("--weight_decay", type=int, default=1e-5)
    parser.add_argument("--variational_beta", type=int, default=1)
    parser.add_argument("--color_channels", type=int, default=1)
    parser.add_argument("--dataset", default="mnist", help="mnist = MNIST, fashion-mnist = FashionMNIST,"
                                                           "cifar = CIFAR10, stl = STL10")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--encoder", default="encoder", help="encoder=Encoder, gaborencoder=GaborEncoder, "
                                                             "gabor2encoder=Gabor2Encoder, gabor3encoder=Gabor3Encoder")

    return parser.parse_args()


def main():

    args = arguments()
    wandb.init(entity="predictive-analytics-lab", project="VAE", config=args)

    c = args.capacity
    latent_dims = args.latent_dims
    dataset = args.dataset

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Dataset
    if args.dataset == "mnist":
        train = MNIST(root="./data", train=True, transform=data_transform, download=True)
        test = MNIST(root="./data", train=False, transform=data_transform, download=True)
    elif args.dataset == "fashion-mnist":
        train = FashionMNIST(root="./data", train=True, transform=data_transform, download=True)
        test = FashionMNIST(root="./data", train=False, transform=data_transform, download=True)
    elif args.dataset == "cifar":
        train = CIFAR10(root="./data", train=True, transform=data_transform, download=True)
        test = CIFAR10(root="./data", train=False, transform=data_transform, download=True)
    elif args.dataset == "stl":
        train = STL10(root="./data", split="unlabeled", transform=data_transform, download=True)
        test = STL10(root="./data", split="test", transform=data_transform, download=True)
    print(f"Dataset: {args.dataset}")

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Colour Channels
    if args.dataset in ["mnist", "fashion-mnist"]:
        color_channels = 1
    else:
        color_channels = 3

    # Image Resolution
    """
    mnist, fashion-mnist: 28->14->7
    cifar: 32->16->8
    stl: 96->48->24
    """
    if args.dataset in ["mnist", "fashion-mnist"]:
        encoder_out_size = 7 * 7
    elif args.dataset in ["cifar"]:
        encoder_out_size = 8 * 8
    elif args.dataset in ["stl"]:
        encoder_out_size = 24 * 24

    encoder_type = args.encoder
    vae = VAE(encoder_type, color_channels, c, encoder_out_size, latent_dims)

    vae = vae.to(device)
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    summary(vae.encoder.features, (3, 32, 32))
    summary(vae.decoder.deconvolution, (512, 4, 4))

    optimizer = optim.Adam(params=vae.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # set to training mode
    vae.train()

    train_loss_avg = []

    print("Training ...")
    for epoch in range(args.epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, _ in train_loader:
            image_batch = image_batch.to(device)

            # VAE Reconstruction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

            # Reconstruction Error
            loss = VAE.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # One step of the optimiser (using the gradients from backpropagation)
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] += loss.item()
        num_batches += 1

        train_loss_avg[-1] /= num_batches
        print(f"Epoch [{epoch+1} / {args.epochs}] average reconstruction error: {train_loss_avg[-1]}")
        train_steps = len(train_loader) * (epoch + 1)
        wandb.log({"Average Reconstruction Error": train_loss_avg[-1]}, step=train_steps)

    # Set to evaluation mode
    vae.eval()

    test_loss_avg, num_batches = 0, 0
    for image_batch, _ in test_loader:

        with torch.no_grad():

            image_batch = image_batch.to(device)

            # VAE Reconstruction
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

            # Reconstruction Error
            loss = VAE.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            test_loss_avg += loss.item()
            num_batches += 1

    test_loss_avg /= num_batches
    print(f"Average reconstruction error: {test_loss_avg}")
    train_steps = len(train_loader) * (epoch + 1)
    wandb.log({"Test Average Reconstruction Error": test_loss_avg}, step=train_steps)

    ################################# Reconstruction Visualisation #####################################################
    vae.eval()
    """
    This function takes as an input the images to reconstruct and the name of the model with which the reconstructions 
    are performed. 
    """
    plt.ion()

    def to_img(x):
        x = x.clamp(0, 1)
        return x

    def show_image(img):
        img = to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def visualise_output(images, model):

        with torch.no_grad():
            images = images.to(device)
            images, _, _ = model(images)
            images = images.cpu()
            images = to_img(images)
            np_imagegrid = torchvision.utils.make_grid(images[1: 50], 10, 5).numpy()
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.show()

    images, labels = iter(test_loader).next()

    # Original image visualisation
    print("Original Image")
    show_image(torchvision.utils.make_grid(images[1: 50], 10, 5))
    plt.show()
    plt.savefig("Original_Image", bbox_inches='tight')
    wandb.save('Original_Image.png')

    # Reconstruct and visualise the images using the VAE
    print("VAE Reconstruction")
    visualise_output(images, vae)
    plt.savefig("VAE_Reconstruction", bbox_inches='tight')
    wandb.save('VAE_Reconstruction.png')

    # Interpolate in Latent Space
    def interpolation(lambda1, model, img1, img2):

        with torch.no_grad():
            # latent vector of first image
            img1 = img1.to(device)
            latent_1, _ = model.encoder(img1)

            # latent vector of second image
            img2 = img2.to(device)
            latent_2, _ = model.encoder(img2)

            # interpolation of the two latent vectors
            inter_latent = lambda1 * latent_1 + (1 - lambda1) * latent_2

            # reconstruct interpolated image
            inter_image = model.decoder(inter_latent)
            inter_image = inter_image.cpu()

            return inter_image

    # sort part of test set by digit
    digits = [[] for _ in range(10)]
    for img_batch, label_batch in test_loader:
        for i in range(img_batch.size(0)):
            digits[label_batch[i]].append(img_batch[i:i + 1])
        if sum(len(d) for d in digits) >= 1000:
            break;

    # interpolation lambdas
    lambda_range = np.linspace(0, 1, 10)

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    for ind, l in enumerate(lambda_range):
        inter_image = interpolation(float(l), vae, digits[7][0], digits[1][0])

        inter_image = to_img(inter_image)

        image = inter_image.numpy()

        axs[ind].imshow(image[0, 0, :, :], cmap='gray')
        axs[ind].set_title('lambda_val=' + str(round(l, 1)))
    plt.show()
    plt.savefig("Interpolate_in_Latent_Space", bbox_inches='tight')
    wandb.save('Interpolate_in_Latent_Space.png')

    # Sample Latent Vector from Prior (VAE as Generator)
    with torch.no_grad():

        # sample latent vectors from the normal distribution
        latent = torch.randn(128, latent_dims, device=device)

        # reconstruct images from the latent vectors
        img_recon = vae.decoder(latent)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(5, 5))
        show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5))
        plt.show()
        plt.savefig("Sample_Latent_Vector_from_Prior.png", bbox_inches='tight')
        wandb.save('Sample_Latent_Vector_from_Prior.png')

    # Show 2D Latent Space
    # load a network that was trained with a 2d latent space
    if latent_dims != args.latent_dims:
        print('Please change the parameters to two latent dimensions.')

    with torch.no_grad():

        # create a sample grid in 2d latent space
        latent_x = np.linspace(-1.5, 1.5, 20)
        latent_y = np.linspace(-1.5, 1.5, 20)
        latents = torch.FloatTensor(len(latent_y), len(latent_x), args.latent_dims)
        for i, lx in enumerate(latent_x):
            for j, ly in enumerate(latent_y):
                latents[j, i, 0] = lx
                latents[j, i, 1] = ly
        latents = latents.view(-1, args.latent_dims)  # flatten grid into a batch

        # reconstruct images from the latent vectors
        latents = latents.to(device)
        image_recon = vae.decoder(latents)
        image_recon = image_recon.cpu()

        fig, ax = plt.subplots(figsize=(10, 10))
        show_image(torchvision.utils.make_grid(image_recon.data[:400], 20, 5))
        plt.show()
        plt.savefig("2DLatent_Space", bbox_inches='tight')
        wandb.save('2DLatent_Space.png')

    def get_representations(model, iterator, device):

        model.eval()

        outputs = []
        intermediates = []
        labels = []

        with torch.no_grad():
            for x, y in iterator:
                x = x.to(device)
                y_pred = model(x)
                h = model(x)

                outputs.append(y_pred.cpu())
                intermediates.append(h.cpu())
                labels.append(y)

        outputs = torch.cat(outputs, dim=0)
        intermediates = torch.cat(intermediates, dim=0)
        labels = torch.cat(labels, dim=0)

        return outputs, intermediates, labels

    def get_pca(data, n_components=2):
        pca = decomposition.PCA()
        pca.n_components = n_components
        pca_data = pca.fit_transform(data)
        return pca_data

    def plot_representations(data, labels, classes, type, n_images=None):
        if n_images is not None:
            data = data[:n_images]
            labels = labels[:n_images]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
        handles, labels = scatter.legend_elements()
        legend = ax.legend(handles=handles, labels=classes)
        if type == "PCA":
            fig.savefig("PCA", bbox_inches='tight')
        else:
            fig.savefig("TSNE", bbox_inches='tight')

    def get_tsne(data, n_components=2, n_images=None):
        if n_images is not None:
            data = data[:n_images]

        tsne = manifold.TSNE(n_components=n_components, random_state=0)
        tsne_data = tsne.fit_transform(data)
        return tsne_data

    # Principle Components Analysis (PCA)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
    outputs, intermediates, labels = get_representations(vae, train_loader, device)

    output_pca_data = get_pca(outputs)
    plot_representations(output_pca_data, labels, classes, "PCA")
    wandb.save('PCA.png')

    # t-Distributed Stochastic Neighbor Embedding (t-SNE)
    n_images = 10_000

    output_tsne_data = get_tsne(outputs, n_images=n_images)
    plot_representations(output_tsne_data, labels, classes, "TSNE", n_images=n_images)
    wandb.save('TSNE.png')


if __name__ == "__main__":
    main()