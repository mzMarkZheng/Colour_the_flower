import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches


from dataset import FlowerDataset
from discriminator import Discriminator
from generator import Generator
from utils import save_some_examples, save_checkpoint, load_checkpoint
import config


def train_func(disc, gen, loader, optim_disc, optim_gen, l1_loss, bce, g_scaler, d_scaler, epoch_no
               ):

    # tqdm is for progress bar
    loop = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        # x is bw image, y is colour image
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Many functions only work with float data type
        x = x.float()
        y = y.float()

        # Train the discriminator
        with torch.cuda.amp.autocast():

            y_fake = gen(x)

            # Loss from fake image
            D_fake = disc(y, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))

            # Loss from real image
            D_real = disc(y, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))

            # Some sources says that the discriminator trains too fast compared to the generator, so it is halved
            # Another source did not have it halved
            D_loss = (D_real_loss + D_fake_loss) / 2

        if torch.sigmoid(D_fake).mean().item() > 0.35:
            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(optim_disc)
            d_scaler.update()

        # Train the generator
        with torch.cuda.amp.autocast():
            D_fake = disc(y, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))

            # L1 is a loss function (least absolute deviations)
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        optim_gen.zero_grad()
        g_scaler.scale(G_loss).backward()

        g_scaler.step(optim_gen)
        g_scaler.update()

        if idx == 0:
            print(
                f"Epoch [{epoch_no}/{config.NUM_EPOCHS}] Batch {idx}/{len(loader)} \
                      Loss D: {D_loss:.4f}, loss G: {G_loss:.4f}"
            )

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=1, out_channels=3).to(config.DEVICE)

    # Standard values for Adam beta 1 is 0.9, but paper has 0.5

    # Optimiser for the discriminator and generator
    optim_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optim_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Loss
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # Load model if we have LOAD_MODEL set to True in configs
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC, disc,
                        optim_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN, gen, optim_gen,
                        config.LEARNING_RATE)

    # Load training dataset
    train_dataset = FlowerDataset(config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)

    # float16 training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Load the testing/validation dataset
    test_dataset = FlowerDataset(config.TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        print(f'====\nEpoch {epoch}\n====')

        train_func(
            disc, gen, train_loader, optim_disc, optim_gen, L1_LOSS, BCE, g_scaler, d_scaler, epoch,
        )

        if config.SAVE_MODEL and epoch % config.SAVE_MODEL_EVERY_NTH == 0:
            save_checkpoint(gen, optim_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, optim_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, test_loader, epoch, folder=config.EXAMPLE_DIR)

    # If by chance epoch number is not divisible, we save the very last epoch
    if config.SAVE_MODEL and epoch % config.SAVE_MODEL_EVERY_NTH != 0:
        save_checkpoint(gen, optim_gen, filename=config.CHECKPOINT_GEN)
        save_checkpoint(disc, optim_disc, filename=config.CHECKPOINT_DISC)
