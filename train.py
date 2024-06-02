import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import Pixel2StateNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)
BATCH_SIZE = 32  
NUM_EPOCHS = 20


def set_seed(seed) -> None:
    '''
    Sets the seed for the environment for reproducibility.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(seed=42)

    # Loading data
    dataset_path_and_file = "dataset/augmented_camera_view/proprio_pixel_dataset-100k_2024-06-02_17-44-33.npz" 

    print("Loading dataset...")
    dataset = np.load(dataset_path_and_file, allow_pickle=True)
    dataset_images = dataset['frames']
    dataset_proprios = dataset['observations']

    # Converting to pandas dataframe 
    print("Converting to pandas dataframe")
    data = {
        'image': list(dataset_images),
        'state_space': list(dataset_proprios)
    }
    dataset_df = pd.DataFrame(data)
    # print(dataset_df.head(5))