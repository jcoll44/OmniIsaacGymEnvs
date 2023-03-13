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

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = True
    render = not headless

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

    #for confidence estimate
    experience = Experience(prior_alpha = 0.0, prior_beta=0.0, length_scale=0.7, num_env = cfg.num_envs)

    while env._simulation_app.is_running() and env.sim_frame_count<1000:
        if env._world.is_playing():
            if env._world.current_time_step_index == 0:
                obs = env._world.reset(soft=True)
            obs = env._task.get_observations()["jackal_view"]["obs_buf"]
            obs = obs.view(cfg.num_envs, -1)
            # actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
            actions = agent.get_action(obs)
            # actions = actions.unsqueeze(0)
            env._task.pre_physics_step(actions)
            env._world.step(render=render)
            obs_buf, rew_buf, reset_buf, extras  = env._task.post_physics_step()
            experience.add_step(obs.cpu().detach().numpy(),rew_buf.cpu().detach().numpy(),reset_buf.cpu().detach().numpy())
            env.sim_frame_count += 1
        else:
            env._world.step(render=render)

    env._simulation_app.close()

    # 0.0, -2.0
    # -1.5, 1.5
    # -0.3, 0.2

    print(len(experience.successful_states))
    x_init_space = np.linspace(-1.5,1.5, 10)
    y_init_space = np.linspace(-2.3,1.8, 10)
    X, Y = np.meshgrid(x_init_space, y_init_space)
    Z = np.zeros_like(X)
    for i,x in enumerate(x_init_space):
        for j,y in enumerate(y_init_space):
            state = np.array([x,y, 0.0635, 0.70711 , 0., 0., 0.70711, -1.2, 1.5])
            Z[i,j], sigma, alpha, beta = experience.get_state_value(state)
    # print(value, sigma, alpha, beta)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()
    plt.savefig("mygraph.png")


if __name__ == '__main__':
    parse_hydra_configs()

