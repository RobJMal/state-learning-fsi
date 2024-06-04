import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import json
from torchsummary import summary
import contextlib
from datetime import datetime
import argparse

from model import Pixel2StateNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)
BATCH_SIZE = 32  
NUM_EPOCHS = 100
SEED = 0


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
    print(f"Loading dataset {dataset_path_and_file}")
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
    # Parsing arguments 
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="dmc-walker-walk.yml", help="config file to run(default: dmc-walker-walk.yml)")
    # parser.add_argument("--log_to_wandb", type=str, default='offline', help="logging to wandb (options: online, offline)")
    args = parser.parse_args()

    # For logging and data collection purposes 
    current_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_directory = f'results/pixel2statenet_training_{current_datetime_str}'
    os.makedirs(results_directory, exist_ok=True)

    set_seed(seed=SEED)

    # Loading data
    dataset_directory = "dataset/augmented_camera_view" 
    dataset_filename = "proprio_pixel_dataset-100k_2024-06-02_17-44-33.npz"
    dataset_path = os.path.join(dataset_directory, dataset_filename)
    dataset_df = load_datset(dataset_path)
    
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

    train_dataset = CustomDataset(images_train, state_space_train)
    test_dataset = CustomDataset(images_test, state_space_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    metadata = {
        'datetime': current_datetime_str,
        'dataset': dataset_filename,
        'device': str(DEVICE),
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'seed': SEED,
        'optimizer': 'Adam',  # Change this if using SGD
        'learning_rate': 1e-4,
        'training_losses': [],
        'validation_losses': [],
        'training_mae': [],
        'validation_mae': [],
    }

    model = Pixel2StateNet().to(DEVICE)

    model_summary_filename = f"model_summary_{current_datetime_str}.txt"
    model_summary_path = os.path.join(results_directory, model_summary_filename)
    with open(model_summary_path, "w") as f:
        with contextlib.redirect_stdout(f):
            summary(model, input_size=(3, 128, 128))
    print(f"Model information saved to {model_summary_path}")

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []
    train_mae, val_mae = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss_train, epoch_mae_train = 0, 0

        # Training loop 
        for images, states in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, states = images.to(DEVICE), states.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, states)
            loss.backward()
            optimizer.step()

            # Calculating metrics 
            epoch_loss_train += loss.item() * images.size(0)
            mae = torch.abs(outputs - states).mean().item()
            epoch_mae_train += mae * images.size(0)

        train_losses.append(epoch_loss_train / len(train_loader.dataset))
        train_mae.append(epoch_mae_train / len(train_loader.dataset))

        # Validation 
        model.eval()
        epoch_loss_val, epoch_mae_val = 0, 0

        with torch.no_grad():
            for images, states in tqdm(test_loader, desc=f"Validation Epoch {epoch+1}/{NUM_EPOCHS}"):
                images, states = images.to(DEVICE), states.to(DEVICE)
                outputs = model(images)
                loss = loss_function(outputs, states)

                # Calculating metrics 
                epoch_loss_val += loss.item() * images.size(0)
                mae = torch.abs(outputs - states).mean().item()
                epoch_mae_val += mae * images.size(0)
        
        val_losses.append(epoch_loss_val / len(test_loader.dataset))
        val_mae.append(epoch_mae_val / len(test_loader))

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Training MAE: {train_mae[-1]:.4f}, Validation MAE: {val_mae[-1]:.4f}")

        # Update metadata
        metadata['training_losses'].append(train_losses[-1])
        metadata['validation_losses'].append(val_losses[-1])
        metadata['training_mae'].append(train_mae[-1])
        metadata['validation_mae'].append(val_mae[-1])

    # Optionally save the model
    model_filename = f"pixel2statenet_model_weights_{current_datetime_str}.pth"
    model_path = os.path.join(results_directory, model_filename)
    torch.save(model.state_dict(), model_path)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(NUM_EPOCHS), train_losses, label='Training Loss')
    plt.plot(range(NUM_EPOCHS), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(NUM_EPOCHS), train_mae, label='Training MAE')
    plt.plot(range(NUM_EPOCHS), val_mae, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    metrics_plot_filename = f"pixel2state_model_metrics_{current_datetime_str}.png"
    metrics_plot_path = os.path.join(results_directory, metrics_plot_filename)
    plt.savefig(metrics_plot_path)

    metadata_filename = f"training_metadata_{current_datetime_str}.json"
    metadata_path = os.path.join(results_directory, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)