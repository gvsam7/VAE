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
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
from torch.utils.data import DataLoader
import argparse
from models.VAE import VAE
from models.Autoencoder import Autoencoder
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
    parser.add_argument("--colour_channels", type=int, default=3)
    parser.add_argument("--dataset", default="cifar", help="mnist = MNIST, fashion-mnist = FashionMNIST,"
                                                           "cifar = CIFAR10, stl = STL10")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", default="vae", help="vae=VAE, autoencoder=Autoencoder")
    parser.add_argument("--encoder", default="encoder", help="encoder=Encoder, gaborencoder=GaborEncoder, "
                                                             "gabor2encoder=Gabor2Encoder, gabor3encoder=Gabor3Encoder,"
                                                             "gabencoder=GabEncoder")

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
        colour_channels = 1
    else:
        colour_channels = 3

    # Image Resolution
    """
    mnist, fashion-mnist: 28->14->7->3
    cifar: 32->16->8->4
    stl: 96->48->24->12
    """
    if args.dataset in ["mnist", "fashion-mnist"]:
        encoder_out_size = 3 * 3
        img_size = 24
        out = 24//2//2//2
    elif args.dataset in ["cifar"]:
        encoder_out_size = 4*4
        img_size = 32
        out = 32//2//2//2
    elif args.dataset in ["stl"]:
        encoder_out_size = 12*12
        img_size = 96
        out = 96//2//2//2
        print(f"out: {out}")

    encoder_type = args.encoder
    # model
    if args.model == 'vae':
        model = VAE(encoder_type, colour_channels, c, encoder_out_size, latent_dims)
    else:
        model = Autoencoder(encoder_type, colour_channels, c, encoder_out_size, latent_dims)
    print(f"model is {args.model}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    summary(model.encoder.features, (colour_channels, img_size, img_size))
    summary(model.decoder.deconvolution, (512, out, out))

    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # set to training mode
    model.train()

    train_loss_avg = []

    print("Training ...")
    for epoch in range(args.epochs):
        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, _ in train_loader:
            image_batch = image_batch.to(device)

            # VAE/AE Reconstruction
            if args.model == 'vae':
                image_batch_recon, latent_mu, latent_logvar = model(image_batch)
            else:
                image_batch_recon = model(image_batch)

            # Reconstruction Error
            if args.model == 'vae':
                loss = VAE.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
            else:
                loss = F.binary_cross_entropy(image_batch_recon, image_batch)

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
    model.eval()

    test_loss_avg, num_batches = 0, 0
    for image_batch, _ in test_loader:

        with torch.no_grad():

            image_batch = image_batch.to(device)

            # AE/VAE Reconstruction
            if args.model == 'vae':
                image_batch_recon, latent_mu, latent_logvar = model(image_batch)
            else:
                image_batch_recon = model(image_batch)

            # Reconstruction Error
            if args.model == 'vae':
                loss = VAE.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
            else:
                loss = F.binary_cross_entropy(image_batch_recon, image_batch)

            test_loss_avg += loss.item()
            num_batches += 1

    test_loss_avg /= num_batches
    print(f"Average reconstruction error: {test_loss_avg}")
    train_steps = len(train_loader) * (epoch + 1)
    wandb.log({"Test Average Reconstruction Error": test_loss_avg}, step=train_steps)

    ################################# Reconstruction Visualisation #####################################################
    model.eval()
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
            if args.model == 'vae':
                images, _, _ = model(images)
            else:
                images=model(images)
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
    visualise_output(images, model)
    plt.savefig("VAE_Reconstruction", bbox_inches='tight')
    wandb.save('VAE_Reconstruction.png')

    # Interpolate in Latent Space
    def interpolation(lambda1, model, img1, img2):

        with torch.no_grad():
            # latent vector of first image
            img1 = img1.to(device)
            if args.model == 'vae':
                latent_1, _ = model.encoder(img1)
            else:
                latent_1 = model.encoder(img1)

            # latent vector of second image
            img2 = img2.to(device)
            if args.model == 'vae':
                latent_2, _ = model.encoder(img2)
            else:
                latent_2 = model.encoder(img2)

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
        inter_image = interpolation(float(l), model, digits[7][0], digits[1][0])

        inter_image = to_img(inter_image)

        image = inter_image.numpy()

        axs[ind].imshow(image[0, 0, :, :], cmap='gray')
        axs[ind].set_title('lambda_val=' + str(round(l, 1)))
    plt.show()
    plt.savefig("Interpolate_in_Latent_Space", bbox_inches='tight')
    wandb.save('Interpolate_in_Latent_Space.png')

    # Sample Latent Vector from Prior (VAE as Generator)
    if args.model == 'vae':
        with torch.no_grad():

            # sample latent vectors from the normal distribution
            latent = torch.randn(128, latent_dims, device=device)

            # reconstruct images from the latent vectors
            img_recon = model.decoder(latent)
            img_recon = img_recon.cpu()

            fig, ax = plt.subplots(figsize=(5, 5))
            show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5))
            plt.show()
            plt.savefig("Sample_Latent_Vector_from_Prior.png", bbox_inches='tight')
            wandb.save('Sample_Latent_Vector_from_Prior.png')
    else:
        None

    # Random Latent Vector (Autoencoder as Generator)
    if args.model == 'autoencoder':
        with torch.no_grad():
            # approx. fit a multivariate Normal distribution (with diagonal cov.) to the latent vectors of a random
            # part of the test set.
            images, labels = iter(test_loader).next()
            images = images.to(device)
            latent = model.encoder(images)
            latent = latent.cpu()

            mean = latent.mean(dim=0)
            std = (latent - mean).pow(2).mean(dim=0).sqrt()

            # sample latent vectors from the normal distribution
            latent = torch.randn(128, 512) * std + mean  # changed latent_dims to 512

            # reconstruct images from the latent vectors
            latent = latent.to(device)
            img_recon = model.decoder(latent)
            img_recon = img_recon.cpu()

            fig, ax = plt.subplots(figsize=(5, 5))
            show_image(torchvision.utils.make_grid(img_recon[:100], 10, 5))
            plt.show()
            print("saving random latent vector...")
            plt.savefig("Random_Latent_Vector", bbox_inches='tight')
            wandb.save('Random_Latent_Vector.png')
    else:
        None

    # Show 2D Latent Space
    # load a network that was trained with a 2d latent space
    if args.model == 'vae':
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
            image_recon = model.decoder(latents)
            image_recon = image_recon.cpu()

            fig, ax = plt.subplots(figsize=(10, 10))
            show_image(torchvision.utils.make_grid(image_recon.data[:400], 20, 5))
            plt.show()
            plt.savefig("2DLatent_Space", bbox_inches='tight')
            wandb.save('2DLatent_Space.png')
    else:
        None


if __name__ == "__main__":
    main()