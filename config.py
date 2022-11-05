# For transformation stuff
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
TRAIN_DIR = "Data/train"
TEST_DIR = "Data/test"
# TRAIN_DIR = "quick_data/train"
# TEST_DIR = "quick_data/test"
EXAMPLE_DIR = "Example_results"

# The paper uses a batch size of 1
BATCH_SIZE = 1
NUM_WORKERS = 0
IMAGE_SIZE = 512
CHANNELS_IMG = 3
NUM_EPOCHS = 500
L1_LAMBDA = 100
LOAD_MODEL = False
SAVE_MODEL = True
SAVE_MODEL_EVERY_NTH = 5
CHECKPOINT_DISC = "Checkpoint/discriminator.pth.tar"
CHECKPOINT_GEN = "Checkpoint/generator.pth.tar"
