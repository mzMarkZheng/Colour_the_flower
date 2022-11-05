import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms, models


class FlowerDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        # for some reason colab created this folder inside train and/or test
        self.list_of_files = [dir for dir in os.listdir(
            self.dir) if dir != ".ipynb_checkpoints"]

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, index):
        # get actual file path
        filename = self.list_of_files[index]
        path = os.path.join(self.dir, filename)

        # load initial image with both coloured and gray image.
        image = Image.open(path)

        # Get the coloured image by converting it to a numpy array and subsetting.
        image_colour = np.array(image)[:, :512, :]

        #print('converting to tensor')
        to_tensor = transforms.ToTensor()
        image_colour = to_tensor(image_colour)

        #print('greyscaling bw image')
        bw_transform = transforms.Grayscale()

        image_bw = bw_transform(image_colour)

        normalise = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        image_colour = normalise(image_colour)

        # print('Done')
        return (image_bw, image_colour)
