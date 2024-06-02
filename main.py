import os 
import copy 

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'    # Connecting to GPU

# Main imports 
from dm_control import suite 
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime

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


if __name__ == '__main__':

    # Ensure the directories exists
    os.makedirs("media", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    domain_name = "fish"
    task_name = "swim"
    seeds = [0, 42, 69, 37, 15, 92, 64, 32, 77, 31]
    camera_view_height, camera_view_width = 64, 64 

    frames_dataset = []
    observations_dataset = []

    for seed in seeds:
        print(f"Generating dataset using seed {seed}")
        random_state = np.random.RandomState(seed)    # Setting the seed 
        env = suite.load(domain_name, task_name, task_kwargs={'random': random_state})
        counter = 0

        spec = env.action_spec()
        time_step = env.reset()

        while not(time_step.last()):
            action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
            time_step = env.step(action)

            camera0 = env.physics.render(camera_id=0, height=camera_view_height, width=camera_view_width)
            camera1 = env.physics.render(camera_id=1, height=camera_view_height, width=camera_view_width)
            camera2 = env.physics.render(camera_id=2, height=camera_view_height, width=camera_view_width)
            camera3 = env.physics.render(camera_id=3, height=camera_view_height, width=camera_view_width)

            top_row = np.concatenate((camera0, camera1), axis=1)
            bottom_row = np.concatenate((camera2, camera3), axis=1)
            camera_obs = np.concatenate((top_row, bottom_row), axis=0)

            frames_dataset.append(camera_obs)
            observations_dataset.append(copy.deepcopy(time_step.observation))
            counter += 1

        print(f"Length of episode: ", counter)
        print(f"Finished dataset using seed {seed}")
        print("")

    frames_dataset_nparray = np.array(frames_dataset)
    observations_dataset_nparray = np.array(observations_dataset)

    print(f"Length of frames_dataset: {len(frames_dataset)}")
    print(f"Length of observations_dataset: {len(observations_dataset)}")

    current_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_filename_and_location = f"dataset/augmented_camera_view/proprio_pixel_dataset_{current_datetime_str}.npz"

    save_data(frames_dataset_nparray, observations_dataset_nparray, dataset_filename_and_location)

