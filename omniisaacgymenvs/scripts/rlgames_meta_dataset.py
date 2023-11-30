'''
This experiment is to use the pretrained agent to analyse in and out of distribution performance for metacognition.
'''
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.demo_util import initialize_demo
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver

from omniisaacgymenvs.utils.dataset import Dataset

import hydra
from omegaconf import DictConfig
import pickle
import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import cv2 as cv2

# Current training range parameters:
# Jackal x -> (-1.5, 1.5)
# Jackal y -> (-0.3, 0.2) or (-2.3, -1.8) because of an offset
# Jackal z_rotation -> (0.0, 3.14)
# Left door x -> (-0.3, 0.3) or (-1.3, -0.7) because of an offset
# Right door x -> (-0.3, 0.3) or (0.7, 1.3) because of an offset
# Noise level -> 0.2

# Training Distribution:
# Jackal x -> (-0.375, 0.375)
# Jackal y -> (-0.1125, 0.0125) or (-2.3, -1.8) because of an offset
# Jackal z_rotation -> (1.1780972451, 1.96349540848)
# Left door x -> (-0.075, 0.075) or (-1.3, -0.7) because of an offset
# Right door x -> (-0.075, 0.075) or (0.7, 1.3) because of an offset
# Noise level -> 0.2

# Test Distribution 1:
# Jackal x -> (-0.75, 0.75)
# Jackal y -> (-0.175, 0.075) or (-2.3, -1.8) because of an offset
# Jackal z_rotation -> (0.78539816341, 2.35619449017)
# Left door x -> (-0.15, 0.15) or (-1.3, -0.7) because of an offset
# Right door x -> (-0.15, 0.15) or (0.7, 1.3) because of an offset
# Noise level -> 0.2

# Test Distribution 2:
# Jackal x -> (-1.125, 1.125)
# Jackal y -> (-0.2375, 0.1375) or (-2.3, -1.8) because of an offset
# Jackal z_rotation -> (0.39269908172, 2.74889357186)
# Left door x -> (-0.225, 0.225) or (-1.3, -0.7) because of an offset
# Right door x -> (-0.225, 0.225) or (0.7, 1.3) because of an offset
# Noise level -> 0.2

# Test Distribution 3:
# Jackal x -> (-1.5, 1.5)
# Jackal y -> (-0.3, 0.2) or (-2.3, -1.8) because of an offset
# Jackal z_rotation -> (0.0, 3.14)
# Left door x -> (-0.3, 0.3) or (-1.3, -0.7) because of an offset
# Right door x -> (-0.3, 0.3) or (0.7, 1.3) because of an offset
# Noise level -> 0.2

def save_video(video, filepath):
    """
    saves a sequence of images as a mpeg video
    """
    # check there is a folder location and if not make it
    isExist = os.path.exists(filepath)
    if not isExist:

    # Create a new directory because it does not exist
        os.makedirs(filepath)
    filename = "vid.mp4"
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath+filename, fourcc, 40.0, (700, 700))

    # print("Saving video")
    for frame in video:
        frame = np.moveaxis(frame, 0, -1)
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)

    # print("Video saved")

    out.release()

def save_first_frame(image, filepath):
    isExist = os.path.exists(filepath)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(filepath)

    image = image.cpu().detach().numpy()[0,:,:,:]
    image = np.moveaxis(image, 0, -1)
    # image = np.moveaxis(image, 0, 1)
    # image = image * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filename = "first_frame.png"
    # im = Image.fromarray(image)
    # im.save(filepath+filename)
    cv2.imwrite(filepath+filename, image)

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = True
    render = True
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    # env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)
    task = initialize_task(cfg_dict, env)
    cfg.num_envs = cfg_dict["task"]["env"]["numEnvs"]
    # cfg.maxEpisodeLength = 500


    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: env
    })


    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    agent = runner.create_player()
    agent.restore(cfg.checkpoint)
    agent.has_batch_dimension = True


    '''
    Loop through the test environments collecting the successes and failures
    '''
    # x_lower_array = [-0.375, -0.75, -1.125, -1.5]
    # x_upper_array = [0.375, 0.75, 1.125, 1.5]
    # y_lower_array = [-0.1125, -0.175, -0.2375, -0.3]
    # y_upper_array = [0.0125, 0.075, 0.1375, 0.2]
    # yaw_lower_array = [1.1780972451, 0.78539816341, 0.39269908172, 0.0]
    # yaw_upper_array = [1.96349540848, 2.35619449017, 2.74889357186, 3.14]
    # left_door_lower_array = [-0.075, -0.15, -0.225, -0.3]
    # left_door_upper_array = [0.075, 0.15, 0.225, 0.3]
    # right_door_lower_array = [-0.075, -0.15, -0.225, -0.3]
    # right_door_upper_array = [0.075, 0.15, 0.225, 0.3]
    x_lower_array = [-0.75, -1.5]
    x_upper_array = [0.75, 1.5]
    y_lower_array = [-0.175, -0.3]
    y_upper_array = [0.075, 0.2]
    yaw_lower_array = [0.78539816341, 0.0]
    yaw_upper_array = [2.35619449017, 3.14]
    left_door_lower_array = [-0.15, -0.3]
    left_door_upper_array = [ 0.15, 0.3]
    right_door_lower_array = [-0.15, -0.3]
    right_door_upper_array = [0.15, 0.3]
    '''
    First we need to collect a dataset to initialize the nonparametric model
    '''
    for environment in range(2):
        print("Environment: ", environment)
        # Create the assessment environment

        # desired number of points
        num_points = 2000

        # file_path = "runs/Jackal/nn/Jackal_"+str(environment+1)+".pth"
        # if cfg.checkpoint:
        #     cfg.checkpoint = retrieve_checkpoint_path(file_path)
        #     if cfg.checkpoint is None:
        #         quit()
        # 


        # Determine the number of points along each axis - first method
        n = round(num_points ** (1/5))
        if n**5 > num_points:
            n = n-1

        # Determine the number of sample points along each axis - second method
        # num_points_x = int(round(num_points ** (1/5)))
        # num_points_y = int(round(num_points ** (1/5)))
        # num_points_yaw = int(round(num_points ** (1/5)))
        # num_points_left_door = int(round(num_points ** (1/5)))
        # num_points_right_door = num_points // (num_points_x * num_points_y * num_points_yaw * num_points_left_door)
        # print("Number of points along each axis: ", num_points_x, num_points_y, num_points_yaw, num_points_left_door, num_points_right_door)

        x_init_space = np.linspace(x_lower_array[environment],x_upper_array[environment], n)
        y_init_space = np.linspace(y_lower_array[environment],y_upper_array[environment], n) # y pos is -2 + this offset
        yaw_init_space = np.linspace(yaw_lower_array[environment],yaw_upper_array[environment], n) 
        left_door_init_space = np.linspace(left_door_lower_array[environment],left_door_upper_array[environment], n) #left door x is -1 + this offset
        right_door_init_space = np.linspace(right_door_lower_array[environment],right_door_upper_array[environment], n) # right door x is 1 + this offset
                
        
        X, Y, YAW, LEFT, RIGHT = np.meshgrid(x_init_space, y_init_space, yaw_init_space, left_door_init_space, right_door_init_space)

        X = X.flatten()
        Y = Y.flatten()
        YAW = YAW.flatten()
        LEFT = LEFT.flatten()
        RIGHT = RIGHT.flatten()

        #add a small amount of gaussian noise to the points
        X += np.random.normal(0, 0.05, size=(X.shape[0]))
        Y += np.random.normal(0, 0.05, size=(Y.shape[0]))
        YAW += np.random.normal(0, 0.05, size=(YAW.shape[0]))
        LEFT += np.random.normal(0, 0.05, size=(LEFT.shape[0]))
        RIGHT += np.random.normal(0, 0.05, size=(RIGHT.shape[0]))
        
        #constrain the points to be within the lower and upper bounds
        X = np.clip(X, x_lower_array[environment], x_upper_array[environment])
        Y = np.clip(Y, y_lower_array[environment], y_upper_array[environment])
        YAW = np.clip(YAW, yaw_lower_array[environment], yaw_upper_array[environment])
        LEFT = np.clip(LEFT, left_door_lower_array[environment], left_door_upper_array[environment])
        RIGHT = np.clip(RIGHT, right_door_lower_array[environment], right_door_upper_array[environment])
        
        number_of_samples = X.shape[0]
        print("Number of samples: ", number_of_samples)

        env.sim_frame_count = 0
        collected_samples = False

        dataset = Dataset(dataset_size=number_of_samples, start_states=np.concatenate((X[:,np.newaxis], Y[:,np.newaxis], YAW[:,np.newaxis], LEFT[:,np.newaxis], RIGHT[:,np.newaxis]), axis=1))
        env._world.step(render=render)
        env._world.step(render=render)
        env._world.step(render=render)

        if environment == 0:
            save_path = "data/id/training/"
        else:
            save_path = "data/id/test/"

        for i in range(X.shape[0]):
            env.sim_frame_count=0
            if i % 100 == 0:
                print(i)
            # 
            video = []
            while env._simulation_app.is_running():
                if env._world.is_playing():
                    if env.sim_frame_count == 0:
                        env._task.set_start_state(np.expand_dims(X[i],0), np.expand_dims(Y[i],0), np.expand_dims(YAW[i],0), np.expand_dims(LEFT[i],0), np.expand_dims(RIGHT[i],0))
                        for j in range(10):
                            env._world.step(render=render)
                    obs = env._task.get_observations()["jackal_view"]["obs_buf"]
                    obs = obs.view(cfg.num_envs, -1)
                    # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
                    actions = agent.get_action(obs)
                    # actions = actions.unsqueeze(0)
                    env._task.pre_physics_step(actions)
                    env._world.step(render=render)
                    obs_buf, rew_buf, reset_buf, extras  = env._task.post_physics_step()
                    collected_samples = dataset.add_step(obs.cpu().detach().numpy(),rew_buf.cpu().detach().numpy(),reset_buf.cpu().detach().numpy())
                    image = env._task.get_image()
                    video.append(image.cpu().detach().numpy()[0,:,:,:])
                    if reset_buf.cpu().detach().numpy()[0] == 1:
                        save_video(video, save_path+str(i)+"/")
                        if rew_buf[0] > 0 and reset_buf[0] == 1:
                            dataset.add_label(1)
                        else:
                            dataset.add_label(0)
                        dataset.add_frame_count(env.sim_frame_count)
                        break
                    if env.sim_frame_count==0:
                        save_first_frame(image, save_path+str(i)+"/")
                    env.sim_frame_count += 1
                else:
                    env._world.step(render=render)  

        success_rate = dataset.get_success_rate()
        print("Success rate: ", success_rate)


        print("Number of Successful states",len(dataset.successful_states))
        print("Number of failure states",len(dataset.unsuccessful_states))


        # Save Training datapoints
        dataset.save(save_path+"dataset.csv", save_path+"metadata.csv")





    env._simulation_app.close()




if __name__ == '__main__':
    parse_hydra_configs()

