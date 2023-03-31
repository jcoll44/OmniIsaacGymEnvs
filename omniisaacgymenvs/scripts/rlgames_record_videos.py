# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
import csv
import cv2
from PIL import Image

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


"""
PYTHON_PATH scripts/rlgames_confidence_ccbp.py task=Jackal headless=True num_envs=100 test=True checkpoint=runs/Jackal/nn/Jackal_discrete.pth 
"""

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = True
    render = True

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)
    task = initialize_task(cfg_dict, env)
    cfg.num_envs = cfg_dict["task"]["env"]["numEnvs"]


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

    # while env._simulation_app.is_running() and env.sim_frame_count<2000:
    #     print(env.sim_frame_count)
    #     if env._world.is_playing():
    #         if env._world.current_time_step_index == 0:
    #             obs = env._world.reset(soft=True)
    #         obs = env._task.get_observations()["jackal_view"]["obs_buf"]
    #         obs = obs.view(cfg.num_envs, -1)
    #         # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
    #         actions = agent.get_action(obs)
    #         # actions = actions.unsqueeze(0)
    #         env._task.pre_physics_step(actions)
    #         env._world.step(render=render)
    #         obs_buf, rew_buf, reset_buf, extras  = env._task.post_physics_step()
    #         env.sim_frame_count += 1
    #     else:
    #         env._world.step(render=render)



    # print(len(experience.successful_states))



    # Create the assessment environment
    x_init_space = np.linspace(-1.5,1.5, 5)
    y_init_space = np.linspace(-0.3, 0.2, 5) # y pos is -2 + this offset
    yaw_init_space = np.linspace(0.0, 3.14, 3) 
    left_door_init_space = np.linspace(-0.2,0.3,3) #left door x is -1 + this offset
    right_door_init_space = np.linspace(-0.3,0.5,3) # right door x is 1 + this offset

    X, Y, YAW, LEFT, RIGHT = np.meshgrid(x_init_space, y_init_space, yaw_init_space, left_door_init_space, right_door_init_space)

    X = X.flatten()
    Y = Y.flatten()
    YAW = YAW.flatten()
    LEFT = LEFT.flatten()
    RIGHT = RIGHT.flatten()

    print(X.shape)
    
    successful_states = np.ones_like(X)
    noise = 0.2
    env._world.step(render=render)
    env._world.step(render=render)
    env._world.step(render=render)
    obs = env._world.reset(soft=True)
    for i in range(X.shape[0]):
        env._task.update_noise_value(noise)
        env.sim_frame_count=0
        # 
        video = []
        while env._simulation_app.is_running():
            if env._world.is_playing():
                if env.sim_frame_count == 0:
                    # obs = env._world.reset(soft=True)
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
                image = env._task.get_image()
                # print(rew_buf)
                # print(reset_buf)
                # print(successful_states)
                video.append(image.cpu().detach().numpy()[0,:,:,:])
                if reset_buf.cpu().detach().numpy()[0] == 1:
                    save_video(video, "recordings/"+str(i)+"/")
                    if rew_buf.cpu().detach().numpy()[0] == 100:
                        # Save csv with video number and whether policy was successful or not
                        with open('recordings/policy_results.csv', mode='a') as file:
                            writer = csv.writer(file)
                            writer.writerow([i, 1, X[i], Y[i], YAW[i], LEFT[i], RIGHT[i]])        
                    else:
                        # Save csv with video number and whether policy was successful or not
                        with open('recordings/policy_results.csv', mode='a') as file:
                            writer = csv.writer(file)
                            writer.writerow([i, 0, X[i], Y[i], YAW[i], LEFT[i], RIGHT[i]])                                              
                    
                    break
                if env.sim_frame_count==0:
                    print("saving first frame")
                    save_first_frame(image, "recordings/"+str(i)+"/")

                
                env.sim_frame_count += 1
            else:
                env._world.step(render=render)    


    env._simulation_app.close()

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

    for frame in video:
        frame = np.moveaxis(frame, 0, -1)
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)


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

if __name__ == '__main__':
    parse_hydra_configs()

