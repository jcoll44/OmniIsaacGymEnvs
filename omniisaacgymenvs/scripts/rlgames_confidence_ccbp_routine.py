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

from omniisaacgymenvs.utils.experience import Experience

import hydra
from omegaconf import DictConfig

import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

"""
PYTHON_PATH scripts/rlgames_confidence_ccbp_routine.py task=Jackal headless=True num_envs=30 test=True checkpoint=runs/Jackal/nn/Jackal.pth enable_livestream=True  
"""

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


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = True
    render = not headless

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)
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
    x_lower_array = [-0.375, -0.75, -1.125, -1.5]
    x_upper_array = [0.375, 0.75, 1.125, 1.5]
    y_lower_array = [-0.1125, -0.175, -0.2375, -0.3]
    y_upper_array = [0.0125, 0.075, 0.1375, 0.2]
    yaw_lower_array = [1.1780972451, 0.78539816341, 0.39269908172, 0.0]
    yaw_upper_array = [1.96349540848, 2.35619449017, 2.74889357186, 3.14]
    left_door_lower_array = [-0.075, -0.15, -0.225, -0.3]
    left_door_upper_array = [0.075, 0.15, 0.225, 0.3]
    right_door_lower_array = [-0.075, -0.15, -0.225, -0.3]
    right_door_upper_array = [0.075, 0.15, 0.225, 0.3]

    '''
    First we need to collect a dataset to initialize the nonparametric model
    '''
    for environment in range(len(x_lower_array)):
        print("Environment: ", environment)
        # Create the assessment environment
        num_points = 100


        # Determine the number of points along each axis
        n = round(num_points ** (1/5))
        if n**5 > num_points:
            n = n-1


        # Determine the number of sample points along each axis
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
        noise = 0.2
        
        X, Y, YAW, LEFT, RIGHT = np.meshgrid(x_init_space, y_init_space, yaw_init_space, left_door_init_space, right_door_init_space)

        X = X.flatten()
        Y = Y.flatten()
        YAW = YAW.flatten()
        LEFT = LEFT.flatten()
        RIGHT = RIGHT.flatten()

        number_of_samples = X.shape[0]
        print("Number of samples: ", number_of_samples)

        assert number_of_samples<=cfg.num_envs, "Number of samples should be less than or equal to number of environments"
        x_start_state = np.zeros((1, cfg.num_envs))
        x_start_state[0, :number_of_samples] = X
        y_start_state = np.zeros((1, cfg.num_envs))
        y_start_state[0, :number_of_samples] = Y
        yaw_start_state = np.zeros((1, cfg.num_envs))
        yaw_start_state[0, :number_of_samples] = YAW
        left_door_start_state = np.zeros((1, cfg.num_envs))
        left_door_start_state[0, :number_of_samples] = LEFT
        right_door_start_state = np.zeros((1, cfg.num_envs))
        right_door_start_state[0, :number_of_samples] = RIGHT



        # Dataset for nonoparametric model for measuring confidence using training environment without noise
        experience = Experience(prior_alpha = 0.0, prior_beta=0.0, length_scale=0.7, num_env = number_of_samples)
        env._task.update_noise_value(noise)

        env.sim_frame_count = 0
        while env._simulation_app.is_running() and env.sim_frame_count<500:
            if env._world.is_playing():
                if env._world.current_time_step_index == 0:
                    env._task.set_start_state(x_start_state, y_start_state, yaw_start_state, left_door_start_state, right_door_start_state)
                    # for j in range(10):
                    #     env._world.step(render=render)
                obs = env._task.get_observations()["jackal_view"]["obs_buf"]
                obs = obs.view(cfg.num_envs, -1)
                # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
                print(obs.shape)
                print(obs)
                actions = agent.get_action(obs)
                # actions = actions.unsqueeze(0)
                env._task.pre_physics_step(actions)
                env._world.step(render=render)
                obs_buf, rew_buf, reset_buf, extras  = env._task.post_physics_step()
                experience.add_step(obs.cpu().detach().numpy()[:number_of_samples,:],rew_buf.cpu().detach().numpy()[:number_of_samples],reset_buf.cpu().detach().numpy()[:number_of_samples])
                env.sim_frame_count += 1
            else:
                env._world.step(render=render)

        print("Number of Successful states",len(experience.successful_states))
        print("Number of failure states",len(experience.unsuccessful_states))

        '''
        Loop through the test environments collecting the successes and failures
        '''
        x_lower_array = [-0.375, -0.75, -1.125, -1.5]
        x_upper_array = [0.375, 0.75, 1.125, 1.5]
        y_lower_array = [-0.1125, -0.175, -0.2375, -0.3]
        y_upper_array = [0.0125, 0.075, 0.1375, 0.2]
        yaw_lower_array = [1.1780972451, 0.78539816341, 0.39269908172, 0.0]
        yaw_upper_array = [1.96349540848, 2.35619449017, 2.74889357186, 3.14]
        left_door_lower_array = [-0.075, -0.15, -0.225, -0.3]
        left_door_upper_array = [0.075, 0.15, 0.225, 0.3]
        right_door_lower_array = [-0.075, -0.15, -0.225, -0.3]
        right_door_upper_array = [0.075, 0.15, 0.225, 0.3]


        for i in range(4):
            print("Test" ,i)

            generated_num_points = 0
            num_points = 100
            test_num_points = num_points

            while generated_num_points<num_points:

                # Determine the number of points along each axis
                n = round(test_num_points ** (1/5))
                if n**5 > test_num_points:
                    n = n-1

                # num_points_x = int(round(test_num_points ** (1/5)))
                # num_points_y = int(round(test_num_points ** (1/5)))
                # num_points_yaw = int(round(test_num_points ** (1/5)))
                # num_points_left_door = int(round(test_num_points ** (1/5)))
                # num_points_right_door = test_num_points // (num_points_x * num_points_y * num_points_yaw * num_points_left_door)

                # Create the assessment environment
                x_init_space = np.linspace(x_lower_array[i],x_upper_array[i], n)
                y_init_space = np.linspace(y_lower_array[i],y_upper_array[i], n) # y pos is -2 + this offset
                yaw_init_space = np.linspace(yaw_lower_array[i],yaw_upper_array[i], n) 
                left_door_init_space = np.linspace(left_door_lower_array[i],left_door_upper_array[i], n) #left door x is -1 + this offset
                right_door_init_space = np.linspace(right_door_lower_array[i],right_door_upper_array[i], n) # right door x is 1 + this offset

                if i>0:
                    # create a boolean mask to identify points that are outside the range of x and y
                    mask = np.logical_or(x_init_space < x_lower_array[i-1], x_init_space > x_upper_array[i-1])
                    # apply the mask to the points array to remove points between x and y
                    x_init_space = x_init_space[mask]

                    # create a boolean mask to identify points that are outside the range of x and y
                    mask = np.logical_or(y_init_space < y_lower_array[i-1], y_init_space > y_upper_array[i-1])
                    # apply the mask to the points array to remove points between x and y
                    y_init_space = y_init_space[mask]        

                    # create a boolean mask to identify points that are outside the range of x and y
                    mask = np.logical_or(yaw_init_space < y_lower_array[i-1], yaw_init_space > yaw_upper_array[i-1])
                    # apply the mask to the points array to remove points between x and y
                    yaw_init_space = yaw_init_space[mask]    

                    # create a boolean mask to identify points that are outside the range of x and y
                    mask = np.logical_or(left_door_init_space < yaw_lower_array[i-1], left_door_init_space > x_upper_array[i-1])
                    # apply the mask to the points array to remove points between x and y
                    left_door_init_space = left_door_init_space[mask]         
                
                    # create a boolean mask to identify points that are outside the range of x and y
                    mask = np.logical_or(right_door_init_space < right_door_lower_array[i-1], right_door_init_space > right_door_upper_array[i-1])
                    # apply the mask to the points array to remove points between x and y
                    right_door_init_space = right_door_init_space[mask]   

                X, Y, YAW, LEFT, RIGHT = np.meshgrid(x_init_space, y_init_space, yaw_init_space, left_door_init_space, right_door_init_space)
        
                generated_num_points = X.flatten().shape[0]
                # print("Number of points generated", generated_num_points)

                test_num_points += 20

        
            X = X.flatten()
            Y = Y.flatten()
            YAW = YAW.flatten()
            LEFT = LEFT.flatten()
            RIGHT = RIGHT.flatten()

            assert generated_num_points<=cfg.num_envs, "Number of samples should be less than or equal to number of environments"
            x_start_state = np.zeros((1, cfg.num_envs))
            x_start_state[0, :generated_num_points] = X
            y_start_state = np.zeros((1, cfg.num_envs))
            y_start_state[0, :generated_num_points] = Y
            yaw_start_state = np.zeros((1, cfg.num_envs))
            yaw_start_state[0, :generated_num_points] = YAW
            left_door_start_state = np.zeros((1, cfg.num_envs))
            left_door_start_state[0, :generated_num_points] = LEFT
            right_door_start_state = np.zeros((1, cfg.num_envs))
            right_door_start_state[0, :generated_num_points] = RIGHT

            environmnents = np.vstack((X,Y,YAW,LEFT,RIGHT)).T

            Prediction = np.zeros_like(X)

            print("Running Predicition")
            for j in range(environmnents.shape[0]):
                state = np.array([X[j], Y[j]-2.0, YAW[j], LEFT[j]-1.0, RIGHT[j]+1.0])
                Prediction[i], sigma, alpha, beta = experience.get_state_value(state)
            print("Completed Prediction")
        
            successful_states = np.zeros_like(X)
            noise = 0.2
            env._task.update_noise_value(noise)
            env.sim_frame_count=0
            obs = env._world.reset(soft=True)
            while env._simulation_app.is_running() and env.sim_frame_count<500:
                if env._world.is_playing():
                    if env._world.current_time_step_index == 0:
                        env._task.set_start_state(x_start_state, y_start_state, yaw_start_state, left_door_start_state, right_door_start_state)

                    obs = env._task.get_observations()["jackal_view"]["obs_buf"]
                    obs = obs.view(cfg.num_envs, -1)
                    # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
                    actions = agent.get_action(obs)
                    # actions = actions.unsqueeze(0)
                    env._task.pre_physics_step(actions)
                    env._world.step(render=render)
                    obs_buf, rew_buf, reset_buf, extras  = env._task.post_physics_step()
                    env.sim_frame_count += 1

                    successful_states += np.where(rew_buf.cpu().detach().numpy()[:generated_num_points] == 100, 1, 0)
                else:
                    env._world.step(render=render)
                
            successful_states = np.where(successful_states > 0, 1, 0)
            print("Actual Success Rate: ",np.sum(successful_states)/generated_num_points)
            success_prediction = np.where(Prediction > 0.5, 1, 0)
            print("Prdicted Success Rate: ",np.sum(success_prediction)/generated_num_points)



    env._simulation_app.close()




if __name__ == '__main__':
    parse_hydra_configs()

