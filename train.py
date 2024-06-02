import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

from model import Pixel2StateNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)
BATCH_SIZE = 32  
NUM_EPOCHS = 20
SEED = 42


def set_seed(seed) -> None:
    '''
    Sets the seed for the environment for reproducibility.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def concatenate_state_space(state_space_dict):
    '''
    Converts the OrderedDict to a single vector for the state
    space. This is for ease of processing when fed into the model. 

    Original dict =  
    data = OrderedDict([
        ('joint_angles', array([7 entries]),
        ('upright', 1 entry),
        ('target', array([3 entries])),
        ('velocity', array([13 entries]))
    ])
    '''
    arrays_list = []
    for key, value in state_space_dict.items():
        if isinstance(value, np.ndarray):
            arrays_list.append(value)
        else:
            arrays_list.append(np.array([value]))

    vector = np.concatenate(arrays_list)
    return vector

def load_datset(dataset_path_and_file="dataset/augmented_camera_view/proprio_pixel_dataset-100k_2024-06-02_17-44-33.npz"):
    '''
    Loads dataset from .npz file. Returns it as a pandas dataframe. 
    '''
    print("Loading dataset")
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

    print("Converting state_space column of dataframe")
    dataset_df['state_space'] = dataset_df['state_space'].apply(lambda x: concatenate_state_space(x))

    return dataset_df

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, state_spaces):
        self.images = images
        self.state_spaces = state_spaces

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        state_space = self.state_spaces[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(state_space, dtype=torch.float32)


if __name__ == "__main__":
    set_seed(seed=SEED)

    # Loading data
    dataset_path_and_file = "dataset/augmented_camera_view/proprio_pixel_dataset-100k_2024-06-02_17-44-33.npz" 
    dataset_df = load_datset(dataset_path_and_file)
    
    # Creating training and test sets
    images_train, images_test, state_space_train, state_space_test = train_test_split(dataset_df['image'].tolist(),
                                                                                    dataset_df['state_space'].tolist(), 
                                                                                    test_size=0.2, 
                                                                                    random_state=SEED)

    # Convert lists back to numpy arrays for ease of use
    images_train = np.array(images_train)
    images_test = np.array(images_test)
    state_space_train = np.array(state_space_train)
    state_space_test = np.array(state_space_test)

    print("Training (images) set size:", images_train.shape)
    print("Training (state_space) set size:", state_space_train.shape)
    print("Test (images) set size:", images_test.shape)
    print("Test (state_space) set size:", state_space_test.shape)