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


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--latent_dims", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--learning_rate", type=int, default=1e-3)
    parser.add_argument("--variational_beta", type=int, default=1)
    parser.add_argument("--color_channels", type=int, default=1)
    parser.add_argument("--dataset", default="mnist", help="mnist = MNIST, fashion-mnist = FashionMNIST,"
                                                           "cifar = CIFAR10, stl = STL10")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)

    return parser.parse_args()


def main():

    args = arguments()

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

    vae = VAE(color_channels, c, encoder_out_size, latent_dims)

    vae = vae.to(device)
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    optimizer = optim.Adam(params=vae.parameters(), lr=args.learning_rate)

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
            loss = VAE.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar, dataset)

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

    # Reconstruct and visualise the images using the VAE
    print("VAE Reconstruction")
    visualise_output(images, vae)


if __name__ == "__main__":
    main()
