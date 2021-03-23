import os

# FOLDER AND FILES
INPUT_DIR = '../input'
TRAIN_FILE = os.path.join(INPUT_DIR, 'train.csv')
TEST_FILE = os.path.join(INPUT_DIR, 'test.csv')
MODEL_DIR = '../models'
FIG_DIR = '../fig'

# DL PARAMS

BATCH_SIZE = 8
EMBED_DIM = 300
HIDDEN_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 10
DEVICE='cpu'
NUM_CLASSES=2