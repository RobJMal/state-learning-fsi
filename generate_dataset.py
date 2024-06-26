import os 

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'    # Connecting to GPU

# Main imports 
from dm_control import suite 
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime
from tqdm import tqdm
import copy 

from utils import parse_args, load_config

# For creating video 
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image


def save_video(frames, filename='media/output.mp4', framerate=30):
    '''
    Creates a video of the provided frames 
    '''
    # Ensure the frames are in the correct shape for matplotlib (height, width, channels)
    frames = [frame.transpose(1, 2, 0) if frame.shape[0] == 3 else frame for frame in frames]

    fig = plt.figure(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100), dpi=100)
    plt.axis('off')

    print("Creating video...")
    ims = [[plt.imshow(frame, animated=True)] for frame in frames]
    ani = animation.ArtistAnimation(fig, ims, interval=1000/framerate, blit=True, repeat_delay=1000)
    ani.save(filename, writer='ffmpeg', fps=framerate)

    print("Finished creating video")


def save_data(frames, observations, filename_and_location='dataset/proprio_pixel_dataset-0.npz'):
    '''
    Saves frames and observations to a compressed .npz file
    '''
    np.savez_compressed(filename_and_location, frames=frames, observations=observations)
    print(f"Data saved to {filename_and_location}")


def generate_episode(seed, frames, observations, target_frame_dim, domain_name="fish", task_name="swim", 
                     camera_view_height=64, camera_view_width=64):
    '''
    Runs a full episode of a domain and task to generate data. Note that it takes in 
    a frames and observations array and it appends to that. 
    '''
    random_state = np.random.RandomState(seed)    # Setting the seed
    env = suite.load(domain_name, task_name, task_kwargs={'random': random_state})

    spec = env.action_spec()
    time_step = env.reset()

    while not time_step.last():
        action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        time_step = env.step(action)

        camera0 = env.physics.render(camera_id=0, height=camera_view_height, width=camera_view_width)
        camera1 = env.physics.render(camera_id=1, height=camera_view_height, width=camera_view_width)
        camera2 = env.physics.render(camera_id=2, height=camera_view_height, width=camera_view_width)
        camera3 = env.physics.render(camera_id=3, height=camera_view_height, width=camera_view_width)

        camera_obs = None

        # Concatentating images in a grid (128, 128, 3)
        if target_frame_dim == (128, 128, 3): 
            top_row = np.concatenate((camera0, camera1), axis=1)
            bottom_row = np.concatenate((camera2, camera3), axis=1)
            camera_obs = np.concatenate((top_row, bottom_row), axis=0)

        # Concatenating images along the axis 
        # Source: https://web.archive.org/web/20170517022842/http://ivpl.eecs.northwestern.edu/sites/default/files/07444187.pdf
        elif target_frame_dim == (64, 64, 12):
            camera_obs = np.concatenate((camera0, camera1, camera2, camera3), axis=2)

        frames.append(camera_obs)
        observations.append(copy.deepcopy(time_step.observation))

    return frames, observations

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    NUM_EPISODES = config['default']['num_episodes']
    BATCH_SIZE = config['default']['batch_size']
    DATASET_DIRECTORY = config['default']['dataset_directory']
    FRAME_SIZE = tuple(config['default']['frame_size'])

    print(f"Generating dataset with {NUM_EPISODES} episodes and {BATCH_SIZE} batch size")
    print(f"Frame size is {FRAME_SIZE}")

    # Ensure the directories exists
    os.makedirs("media", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    os.makedirs(DATASET_DIRECTORY, exist_ok=True)

    domain_name = "fish"
    task_name = "swim"
    seeds = [i for i in range(NUM_EPISODES)]
    camera_view_height, camera_view_width = 64, 64 

    current_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(DATASET_DIRECTORY, exist_ok=True)

    for batch_start in tqdm(range(0, NUM_EPISODES, BATCH_SIZE)):
        batch_end = min(batch_start + BATCH_SIZE, NUM_EPISODES)
        batch_seeds = seeds[batch_start: batch_end]

        frames_dataset = []
        observations_dataset = []

        for seed in tqdm(batch_seeds, total=(len(batch_seeds))):
            print(f"Generating dataset using seed {seed}")
            generate_episode(seed, frames_dataset, observations_dataset, FRAME_SIZE, domain_name, task_name, 
                                camera_view_height, camera_view_width)

        frames_dataset_nparray = np.array(frames_dataset)
        observations_dataset_nparray = np.array(observations_dataset)

        print(f"Length of frames_dataset: {len(frames_dataset)}")
        print(f"Length of observations_dataset: {len(observations_dataset)}")
        
        dataset_filename = f"proprio_pixel_dataset-{NUM_EPISODES}k-start-{batch_start}-end-{batch_end}_{current_datetime_str}.npz"
        dataset_path = os.path.join(DATASET_DIRECTORY, dataset_filename)

        print("Saving data...")
        save_data(frames_dataset_nparray, observations_dataset_nparray, dataset_path)
        print(f"Data batch {batch_start}-{batch_end} saving completed")

