from typing import Tuple, List
from pathlib import Path

# Training Consts
NUM_EPOCHS: int = 10
lr: float = 1e-3
FINE_TUNING_LR: float = 1e-4
BATCH_SIZE: int = 32
PRINT_EVERY: int = 1
SAVE_EVERY: int = 10
CLASSES: List[str] = ['cat', 'dog']
NUM_CLASSES: int = len(CLASSES)
NUM_WORKERS: int = 2
RESNET_INPUT_SIZE: int = 224
RESNET_NUM_LAYERS: int = 18  # 34
RESENT_OUTPUT_DIM: int = 512
MEAN_NORMALIZATION_VECTOR: List[float] = [0.485, 0.456, 0.406]
STD_NORMALIZATION_VECTOR: List[float] = [0.229, 0.224, 0.225]
TRAIN_CLASS_NUM_SAMPLES: int = 800
TEST_CLASS_NUM_SAMPLES: int = 250
# Early Stopping Consts
MIN_IMPROVEMENT: float = 1e-3
PATIENT_NUM_EPOCHS: int = 10

FILES_PATH: str = r"C:\Users\asher\OneDrive\Documents\Data Science Degree\3th Year\Deep Learning\Deep-Learning-Course-TAU\HW2"
