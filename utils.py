import itertools
import torch
import config
from torchvision.utils import save_image


def save_some_examples(gen, val_loader, epoch, folder, num_photos=12):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE, dtype=torch.float), y.to(
        config.DEVICE, dtype=torch.float)

    gen.eval()
    with torch.no_grad():
        index = 0
        label_tensor = None
        output_tensor = None
        for data in itertools.islice(iter(val_loader), num_photos):
            x, y = data
            x, y = x.to(config.DEVICE, dtype=torch.float), y.to(
                config.DEVICE, dtype=torch.float)

            y_fake = gen(x)

            # remove normalization#
            y = y * 0.5 + 0.5
            y_fake = y_fake * 0.5 + 0.5

            if label_tensor is None:
                label_tensor = torch.Tensor(y)
            else:
                # The 3 makes the concatenation happen column-wise.
                label_tensor = torch.cat((label_tensor, y), 3)

            y_fake_denormalized = y_fake

            if output_tensor is None:
                output_tensor = torch.Tensor(y_fake)
            else:
                output_tensor = torch.cat((output_tensor, y_fake), 3)

        save_image(torch.cat((output_tensor, label_tensor), 2),
                   folder + f"/epoch{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
