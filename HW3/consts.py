from typing import Tuple, List
from pathlib import Path
from torch import tensor

SEED: int = 42
# Training Consts
NUM_EPOCHS: int = 100
lr: float = 1e-4
BATCH_SIZE: int = 32
NUM_USERS: int = 6040
PRINT_EVERY: int = 1
SAVE_EVERY: int = 10
ALPHA: float = 2
ALPHA_TENSOR: tensor = tensor(ALPHA)
GAMMA: float = 2
WEIGHT_DECAY: float = 1e-7

# Early Stopping Consts
MIN_IMPROVEMENT: float = 1e-3
PATIENT_NUM_EPOCHS: int = 10

# File Paths
PROJECT_PATH: Path = Path.cwd()
DATA_PATH: Path = Path('Data')
TRAIN_DF_PATH: Path = (DATA_PATH / 'Train.csv').resolve()
TEST_RAND_DF_PATH: Path = (DATA_PATH / 'RandomTest.csv').resolve()
TEST_POP_DF_PATH: Path = (DATA_PATH / 'PopularityTest.csv').resolve()
