import os
from pathlib import Path

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the dataset directory
DATASET_DIR = os.path.join(BASE_DIR, 'ds')

home_dir = str(Path.home())
RESULTS_DIR = os.path.join(home_dir, 'ppcon_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
# Specific dataset paths
# TRAIN_CSV = os.path.join(DATASET_DIR, 'float_ds_sf_train.csv')
# TEST_CSV = os.path.join(DATASET_DIR, 'float_ds_sf_test.csv')
