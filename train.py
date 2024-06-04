import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import json
from torchsummary import summary
import contextlib
from datetime import datetime
import yaml
import argparse

from model import Pixel2StateNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)


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
    max_data = 1000
    dataset_images = dataset['frames'][:max_data]
    dataset_proprios = dataset['observations'][:max_data]

    # Converting to pandas dataframe 
    print("Converting to pandas dataframe")
    data = {
        'image': list(dataset_images),
        'state_space': list(dataset_proprios)
    }
    dataset_df = pd.DataFrame(data)

    print("Converting state_space column of dataframe")
    dataset_df['state_space'] = dataset_df['state_space'].apply(lambda x: concatenate_state_space(x)[:11])
    # dataset_df['state_space'] = dataset_df['state_space'].apply(lambda x: concatenate_state_space(x)[8:11])

    # Normalize the state space data
    # statespace = np.stack(dataset_df['state_space'])
    # mean = statespace.mean(axis=0)
    # std = statespace.std(axis=0)
    # dataset_df['state_space'] = dataset_df['state_space'].apply(lambda x: (x - mean) / std)

    # # Plot histogram of state space
    print("Plotting histogram of state space")
    plt.figure()
    for idx in range(dataset_df['state_space'][0].shape[0]):
        plt.hist([x[idx] for x in dataset_df['state_space']], bins=50, alpha=0.2, color=matplotlib.colormaps['rainbow'](idx/dataset_df['state_space'][0].shape[0]), label=f'Index {idx}')
    plt.xlabel(f'State space')
    plt.ylabel('Frequency')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4)
    # plt.xlim(-1, 1)
    plt.title('Histogram of state space length')
    plt.savefig(f"state_space_histogram.png", bbox_inches='tight')

    # assert False
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

def plot_training_metrics(epoch, train_losses=[], train_mae=[], 
                            val_losses=[], val_mae=[],
                            metrics_plot_filename="pixel2state_model_loss.png"):
    '''
    Plots the training metrics (loss and mean absolute error (MAE) values).
    '''
    plt.figure()

    # Plot for losses
    plt.subplot(2, 1, 1)
    if len(train_losses) > 0:
        plt.plot(range(epoch+1), train_losses, label='Training Loss')
    if len(val_losses) > 0: 
        plt.plot(range(epoch+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()

    # Plot for MAE (Mean Absolute Error)
    plt.subplot(2, 1, 2)
    if len(train_mae) > 0:
        plt.plot(range(epoch+1), train_mae, label='Training MAE')
    if len(val_mae):
        plt.plot(range(epoch+1), val_mae, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(metrics_plot_filename)
    plt.close()

def parse_args():
    '''
    Parses command line arguments for specifying the config file.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.yaml", help="config file to run(default: config_train.yml)")
    return parser.parse_args()

def load_config(config_file):
    '''
    Loads the configuration from a YAML file.
    '''
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    BATCH_SIZE = config['training']['batch_size']
    NUM_EPOCHS = config['training']['num_epochs']
    SEED = config['training']['seed']
    LEARNING_RATE = config['training']['learning_rate']
    STEP_SIZE = config['training']['step_size']
    GAMMA = config['training']['gamma']
    DATASET_DIRECTORY = config['training']['dataset_directory']
    DATASET_FILENAME = config['training']['dataset_filename']

    RESULTS_DIRECTORY = config['logging']['results_directory']

    # For logging and data collection purposes 
    current_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_directory = f'{RESULTS_DIRECTORY}/pixel2statenet_training_{current_datetime_str}'
    os.makedirs(results_directory, exist_ok=True)
    
    set_seed(seed=SEED)

    # Loading data
    dataset_path = os.path.join(DATASET_DIRECTORY, DATASET_FILENAME)
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
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=state_space_train.shape[0], shuffle=False)

    metadata = {
        'datetime': current_datetime_str,
        'dataset': DATASET_FILENAME,
        'device': str(DEVICE),
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'seed': SEED,
        'optimizer': 'Adam',  # Change this if using SGD
        'learning_rate': LEARNING_RATE,
        'step_size': STEP_SIZE,
        'gamma': GAMMA,
        'training_losses': [],
        'validation_losses': [],
        'training_mae': [],
        'validation_mae': [],
    }

    model = Pixel2StateNet().to(DEVICE)
    numParams = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print(f"Number of trainable parameters in model: {numParams}")
    

    model_summary_filename = f"model_summary_{current_datetime_str}.txt"
    model_summary_path = os.path.join(results_directory, model_summary_filename)
    with open(model_summary_path, "w") as f:
        with contextlib.redirect_stdout(f):
            summary(model, input_size=(3, 128, 128))
    print(f"Model information saved to {model_summary_path}")

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    train_losses, val_losses = [], []
    train_mae, val_mae = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss_train, epoch_mae_train = 0, 0

        # Training loop 
        for images, states in train_loader:
        # for images, states in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
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
        scheduler.step()

        # Validation 
        model.eval()
        epoch_loss_val, epoch_mae_val = 0, 0

        with torch.no_grad():
            # for images, states in tqdm(test_loader, desc=f"Validation Epoch {epoch+1}/{NUM_EPOCHS}"):
            for images, states in test_loader:
                images, states = images.to(DEVICE), states.to(DEVICE)
                outputs = model(images)
                loss = loss_function(outputs, states)

                # Calculating metrics 
                epoch_loss_val += loss.item() * images.size(0)
                mae = torch.abs(outputs - states).mean().item()
                epoch_mae_val += mae * images.size(0)
                rel_err = (torch.abs(outputs - states) / torch.abs(states).max(0)[0]).mean().item()

                # Compute distance error
                # dist_err = torch.norm(outputs - states, dim=1).mean().item()
                # rel_dist_err = (torch.norm(outputs - states, dim=1) / torch.norm(states, dim=1)).mean().item()

                # print(f"Distance error: {1e3*dist_err:.4f} mm \t- Relative distance error: {100*rel_dist_err:.2f}%")


                # Plot histogram of state space
                errors = torch.abs(outputs - states).cpu().numpy()
                plt.figure()
                for idx in range(errors.shape[1]):
                    plt.hist(errors[:,idx], bins=20, alpha=0.2, color=matplotlib.colormaps['rainbow'](idx/dataset_df['state_space'][0].shape[0]), label=f'Index {idx}')
                plt.xlabel(f'Error')
                plt.ylabel('Frequency')
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4)
                # plt.xlim(-1, 1)
                plt.savefig(f"error_histogram.png", bbox_inches='tight')
                plt.close()

        
        val_losses.append(epoch_loss_val / len(test_loader.dataset))
        val_mae.append(epoch_mae_val / len(test_loader))

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} with LR {scheduler.get_last_lr()[0]:.2e}, Training Loss: {train_losses[-1]:.4e}, Validation Loss: {val_losses[-1]:.4e}, Training MAE: {train_mae[-1]:.2e}, Validation MAE: {val_mae[-1]:.2e}, Relative error: {100*rel_err:.4f}%")

        # Update metadata
        metadata['training_losses'].append(train_losses[-1])
        metadata['validation_losses'].append(val_losses[-1])
        metadata['training_mae'].append(train_mae[-1])
        metadata['validation_mae'].append(val_mae[-1])

        plot_training_metrics(epoch, train_losses=train_losses, train_mae=train_mae,
                              val_losses=val_losses, val_mae=val_mae)

    # Saving the model
    model_filename = f"pixel2statenet_model_weights_{current_datetime_str}.pth"
    model_path = os.path.join(results_directory, model_filename)
    torch.save(model.state_dict(), model_path)

    metadata_filename = f"training_metadata_{current_datetime_str}.json"
    metadata_path = os.path.join(results_directory, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)