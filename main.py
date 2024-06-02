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


if __name__ == '__main__':

    # Ensure the directories exists
    os.makedirs("media", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    domain_name = "fish"
    task_name = "swim"
    seed = 42
    camera_view_height, camera_view_width = 64, 64 

    random_state = np.random.RandomState(seed)    # Setting the seed 
    env = suite.load(domain_name, task_name, task_kwargs={'random': random_state})


    duration = 4  # Seconds
    frames = []
    ticks = []
    rewards = []
    observations = []

    spec = env.action_spec()
    time_step = env.reset()

    while env.physics.data.time < duration:

        action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        time_step = env.step(action)

        camera0 = env.physics.render(camera_id=0, height=200, width=200)
        camera1 = env.physics.render(camera_id=1, height=200, width=200)
        camera2 = env.physics.render(camera_id=2, height=200, width=200)
        camera3 = env.physics.render(camera_id=3, height=200, width=200)

        top_row = np.concatenate((camera0, camera1), axis=1)
        bottom_row = np.concatenate((camera2, camera3), axis=1)
        camera_obs = np.concatenate((top_row, bottom_row), axis=0)

        frames.append(camera_obs)
        rewards.append(time_step.reward)
        observations.append(copy.deepcopy(time_step.observation))
        ticks.append(env.physics.data.time)

    output_policy_video_filename = f'media/{domain_name}-{task_name}_result.mp4'
    save_video(frames=frames, filename=output_policy_video_filename, framerate=30)

