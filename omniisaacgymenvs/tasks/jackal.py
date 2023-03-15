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


from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.jackal import Jackal
from omniisaacgymenvs.robots.articulations.walls import Walls

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere

import omni

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from pxr import PhysicsSchemaTools, PhysxSchema
from gym import spaces

import numpy as np
import torch
import math

"""
1. Add noise/delayed actions at will. Only 1 delay is required
2. Allow test points to be assessed at will
3. 
"""
"""
Process
1. Train an agent in an environment without noise
2. Assess the metacognitive ability of CCBP at set start states - the dataset for creating the CCBP must be the same as the training one.
3. Save a dataset for training a classifier using the training environment data
4. Assess the classifier at the set start states like CCBP
"""

"""
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host     -v /usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json     -v /usr/share/vulkan/implicit_layer.d/nvidia_layers.json:/etc/vulkan/implicit_layer.d/nvidia_layers.json     -v /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw     -v ~/docker/isaac-sim/config:/root/.nvidia-omniverse/config:rw     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw     -v ~/docker/isaac-sim/documents:/root/Documents:rw  -v ~/Documents/:/home/Documents   nvcr.io/nvidia/isaac-sim:2022.2.0
alias PYTHON_PATH=/isaac-sim/python.sh
cd /home/Documents/OmniIsaacGymEnvs/
PYTHON_PATH -m pip install -e .
PYTHON_PATH -m pip install wandb
cd omniisaacgymenvs
PYTHON_PATH scripts/rlgames_train.py task=Jackal headless=True wandb_activate=True wandb_project=Jackal_Meta wandb_entity=jcoll44
"""


class JackalTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_velocity = 20.0
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self._dt = self._task_cfg["sim"]["dt"]

        self._noise_amount = self._task_cfg["env"]["noiseAmount"]

        self._num_observations = 5
        self._num_actions = 2

        # self.action_space = spaces.Discrete(self._num_actions)



        RLTask.__init__(self, name, env)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.target_position = torch.tensor([0.5, 2.0, 0], device=self.device)

        if self._noise_amount <= self._dt:
            self._action_array = torch.zeros([self._num_envs, 1, 2], dtype=torch.float , device=self.device)
        else:
            self._action_array = torch.zeros([self._num_envs, round(self._noise_amount/self._dt), 2], dtype=torch.float , device=self.device)


        # self.action_space = spaces.Discrete(self._num_actions)


        return

    def set_up_scene(self, scene) -> None:
        self.get_jackal()
        self.get_walls()
        self.get_props()
        # self.create_camera()
        super().set_up_scene(scene)
        self._jackals = ArticulationView(prim_paths_expr="/World/envs/.*/Jackal", name="jackal_view", reset_xform_properties=False)
        self._walls = RigidPrimView(prim_paths_expr="/World/envs/.*/Wall", name="walls_view", reset_xform_properties=False)
        self._left_door = RigidPrimView(prim_paths_expr="/World/envs/.*/left_door", name="left_door", reset_xform_properties=False)
        self._right_door = RigidPrimView(prim_paths_expr="/World/envs/.*/right_door", name="right_door", reset_xform_properties=False)
        scene.add(self._jackals)
        scene.add(self._walls)
        scene.add(self._left_door)
        scene.add(self._right_door)
        # scene.add_ground_plane(size=200.0, color=torch.tensor([0.01,0.01,0.01]))

        self.root_pos, self.root_rot = self._jackals.get_world_poses(clone=False)
        # self.dof_pos = self._jackals.get_joint_positions(clone=False)
        # self.dof_vel = self._jackals.get_joint_velocities(clone=False)
        self.left_root_pos, self.left_root_rot = self._left_door.get_world_poses(clone=False)
        self.right_root_pos, self.right_root_rot = self._right_door.get_world_poses(clone=False)

        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()
        self.initial_left_pos, self.initial_left_rot = self.left_root_pos.clone(), self.left_root_rot.clone()
        self.initial_right_pos, self.initial_right_rot = self.right_root_pos.clone(), self.right_root_rot.clone()

        return

    def get_jackal(self):
        jackal = Jackal(prim_path=self.default_zero_env_path + "/Jackal", name="Jackal", translation=torch.tensor([0.0, -2.0, 0.065]), orientation=torch.tensor([1,0,0,0]))
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Jackal", get_prim_at_path(jackal.prim_path), self._sim_config.parse_actor_config("Jackal"))

    def get_walls(self):
        scene = Walls(prim_path=self.default_zero_env_path + "/Wall", name="Wall", translation=torch.tensor([0.0, 0.0, 0.0]))
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Wall", get_prim_at_path(scene.prim_path), self._sim_config.parse_actor_config("Wall"))

    def get_props(self):
        # left_pos = np.random.uniform(-0.7,-1.2)
        # right_pos = np.random.uniform(0.7,1.5)
        self.left_door = DynamicCuboid(
                prim_path=self.default_zero_env_path + "/left_door", # The prim path of the cube in the USD stage
                name="left_door", # The unique name used to retrieve the object from the scene later on
                position=np.array([-1.0, 0.3, 1.105]), # Using the current stage units which is in meters by default.
                scale=np.array([1.0, 0.5, 2.2]), # most arguments accept mainly numpy arrays.
                color=np.array([1.0, 1.0, 1.0]), # RGB channels, going from 0-1
                mass=1000
            )

        self.right_door = DynamicCuboid(
                prim_path=self.default_zero_env_path + "/right_door", # The prim path of the cube in the USD stage
                name="right_door", # The unique name used to retrieve the object from the scene later on
                position=np.array([1.0, 0.3, 1.105]), # Using the current stage units which is in meters by default.
                scale=np.array([1.0, 0.5, 2.2]), # most arguments accept mainly numpy arrays.
                color=np.array([1.0, 1.0, 1.0]), # RGB channels, going from 0-1
                mass=1000
            )


        self._sim_config.apply_articulation_settings("left_door", get_prim_at_path(self.left_door.prim_path), self._sim_config.parse_actor_config("Left_Door"))
        self._sim_config.apply_articulation_settings("right_door", get_prim_at_path(self.right_door.prim_path), self._sim_config.parse_actor_config("Left_Door"))
        # self.right_door.disable_rigid_body_physics() 
        # self.left_door.disable_rigid_body_physics() 

    # def create_camera(self):
    #     stage = omni.usd.get_context().get_stage()
    #     self.view_port = omni.kit.viewport_legacy.get_default_viewport_window()
    #     # Create camera
    #     self.camera_path = "/World/envs/.*/camera"
    #     self.perspective_path = "/OmniverseKit_Persp"
    #     camera_prim = stage.DefinePrim(self.camera_path, "Camera")
    #     self.view_port.set_active_camera(self.camera_path)
    #     camera_prim.GetAttribute("focalLength").Set(8.5)
    #     self.view_port.set_active_camera(self.perspective_path)

    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._jackals.get_world_poses(clone=False)
        self.root_right_pos, _ = self._right_door.get_world_poses(clone=False)
        self.root_left_pos, _ = self._left_door.get_world_poses(clone=False)

        root_positions = self.root_pos - self._env_pos

        roll,pitch,yaw = get_euler_xyz(self.root_rot)

        right_pos = self.root_right_pos - self.initial_right_pos.clone()
        right_pos[:,0] = right_pos[:,0] +1.0
        left_pos = self.root_left_pos - self.initial_left_pos.clone()
        left_pos[:,0] = left_pos[:,0] -1.0
       
        # root_positions = self.root_pos
        self.obs_buf[..., 0:2] = root_positions[:,:2]
        self.obs_buf[..., 2] = yaw
        self.obs_buf[..., 3] = right_pos[:,0]
        self.obs_buf[..., 4] = left_pos[:,0]

        # torch.set_printoptions(threshold=10_000)
        # print(self.obs_buf) # prints the whole tensor

        observations = {
            self._jackals.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)



        actions = actions.to(self._device)
        actions = torch.where(actions > 1.0, (actions-1)*-1, actions)

        velocity = torch.zeros((self._jackals.count, self._jackals.num_dof), dtype=torch.float32, device=self._device)
        velocity[:,0],velocity[:,1] = self.wheel_velocities(actions[:,0]*2,actions[:,1]*30)
        velocity[:,2],velocity[:,3] = self.wheel_velocities(actions[:,0]*2,actions[:,1]*30)

        indices = torch.arange(self._jackals.count, dtype=torch.int32, device=self._device)
        self._jackals.set_joint_velocity_targets(velocity, indices=indices)



        # # Discretised Actions
        # if actions.ndim > 1: #for some reason the first action is always 2 dimensional...
        #     actions = actions[:,0]
        # else:
        #     actions = actions
        # actions = actions.to(self._device)

        # # From integers to continuous linear and angular velocities
        # body_velocities = torch.zeros((self._jackals.count, 2), dtype=torch.float32, device=self._device)
        # # linear velocity
        # # linear velocity
        # body_velocities[:,0] = torch.where(actions == 0 , 2.0, 0.0)
        # body_velocities[:,0] = torch.where(actions == 1, 1.0, body_velocities[:,0])
        # body_velocities[:,0] = torch.where(actions == 2, 1.0, body_velocities[:,0])
        # body_velocities[:,0] = torch.where(actions == 3 , -2.0, body_velocities[:,0])
        # body_velocities[:,0] = torch.where(actions == 4 , 0.0, body_velocities[:,0])
        # # angular velocity
        # body_velocities[:,1] = torch.where(actions == 1, 30.0, 0.0)
        # body_velocities[:,1] = torch.where(actions == 2, -30.0, body_velocities[:,1])
        # # Save to an array to add noise
        # # self._action_array[:,-1,0] = body_velocities[:,0]
        # # self._action_array[:,-1,1] = body_velocities[:,1]
        # velocity = torch.zeros((self._jackals.count, self._jackals.num_dof), dtype=torch.float32, device=self._device)
        # # velocity[:,0],velocity[:,1] = self.wheel_velocities(self._action_array[:,0,0],self._action_array[:,0,1])
        # # velocity[:,2],velocity[:,3] = self.wheel_velocities(self._action_array[:,0,0],self._action_array[:,0,1])


        # velocity[:,0],velocity[:,1] = self.wheel_velocities(body_velocities[:,0],body_velocities[:,1]) #used when not considering noise
        # velocity[:,2],velocity[:,3] = self.wheel_velocities(body_velocities[:,0],body_velocities[:,1])

        # indices = torch.arange(self._jackals.count, dtype=torch.int32, device=self._device)
        # self._jackals.set_joint_velocity_targets(velocity, indices=indices)

        # self._action_array[:,0:-1,:] = self._action_array[:,1:,:]



    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # rot = torch.zeros(num_resets, dtype=torch.float, device=self.device)
        rand_floats = torch_rand_float(0.0, 3.14, (len(env_ids),1), device=self.device)
        new_jackal_rot = quat_from_angle_axis(rand_floats[:,0], self.z_unit_tensor[env_ids])

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.3, 0.2, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()

        self._jackals.set_velocities(root_velocities[env_ids], indices=env_ids)
        self._jackals.set_world_poses(root_pos[env_ids], new_jackal_rot, indices=env_ids)

        root_pos = self.initial_left_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.2,0.3, (num_resets, 1), device=self._device).view(-1)
        # root_pos[env_ids, 2] += 0.01

        self._left_door.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)

        root_pos = self.initial_right_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.3,0.5, (num_resets, 1), device=self._device).view(-1)
        # root_pos[env_ids, 2] += 0.01

        self._right_door.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)

        # Add *sleep* between actions - the easiest option will be have an action offset I think
        if self._noise_amount <= self._dt:
            action_array = torch.zeros([self._num_envs, 1, 2], dtype=torch.float , device=self.device)
        else:
            action_array = torch.zeros([self._num_envs, round(self._noise_amount/self._dt), 2], dtype=torch.float , device=self.device)
        self._action_array[env_ids,:,:] = action_array[env_ids,:,:]

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.root_velocities = self._jackals.get_velocities(clone=False)
        # self.root_pos, self.root_rot = self._jackals.get_world_poses(clone=False)
        # # self.root_velocities = self._jackals.get_velocities(clone=False)
        # # self.dof_pos = self._jackals.get_joint_positions(clone=False)
        # # self.dof_vel = self._jackals.get_joint_velocities(clone=False)
        # self.left_root_pos, self.left_root_rot = self._left_door.get_world_poses(clone=False)
        # self.right_root_pos, self.right_root_rot = self._right_door.get_world_poses(clone=False)

        # self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()
        # self.initial_left_pos, self.initial_left_rot = self.left_root_pos.clone(), self.left_root_rot.clone()
        # self.initial_right_pos, self.initial_right_rot = self.right_root_pos.clone(), self.right_root_rot.clone()

    def calculate_metrics(self) -> None:
        jackal_pos = self.obs_buf[..., 0:2]

        # cart_pos = self.obs_buf[:, 0]
        # cart_vel = self.obs_buf[:, 1]
        # pole_angle = self.obs_buf[:, 2]
        # pole_vel = self.obs_buf[:, 3]

        dist = ((jackal_pos-self.target_position[0:2])**2).sum(axis=1)
        reward1 = torch.exp(dist/10)*-1
        reward2 = torch.where(torch.abs(dist) < 0.5, 100, reward1)
        reward = reward2


        # reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        jackal_pos = self.obs_buf[..., 0:2]


        dist = ((jackal_pos-self.target_position[0:2])**2).sum(axis=1)

        # pole_pos = self.obs_buf[:, 2]

        resets = torch.where(torch.abs(dist) < 0.5, 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), resets)

        # original code from Raunak on how to calculate termination state
        # xgoal_lo=0.0,xgoal_hi=1.0,ygoal_lo=1.0,ygoal_hi=2.0
        # if bot_yloc >ygoal_lo and bot_yloc<ygoal_hi and bot_xloc>xgoal_lo and bot_xloc<xgoal_hi:
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets


    def wheel_velocities(self, linear_velocity, angular_volicity):

        return (linear_velocity - (0.38/2)*angular_volicity)/0.1, (linear_velocity + (0.38/2)*angular_volicity)/0.1


    def update_noise_value(self, noise_value):
        self._noise_amount = noise_value

        if self._noise_amount <= self._dt:
            action_array = torch.zeros([self._num_envs, 1, 2], dtype=torch.float , device=self.device)
        else:
            action_array = torch.zeros([self._num_envs, round(self._noise_amount/self._dt), 2], dtype=torch.float , device=self.device)

        #Save all the previously commanded values (although this shouldn't be needed as a reset should be done)
        action_array[:,:,:] = self._action_array[:,:action_array.shape[1],:] #second axis is the only axis of change
        self._action_array = action_array

    def set_start_state(self, jackal_x, jackal_y, jackal_yaw, left_door, right_door):
        new_jackal_rot = quat_from_angle_axis(jackal_yaw, self.z_unit_tensor[self._num_envs])

        root_pos = self.initial_root_pos.clone()
        root_pos[:, 0] += jackal_x
        root_pos[:, 1] += jackal_y
        root_velocities = self.root_velocities.clone()

        self._jackals.set_velocities(root_velocities[:])
        self._jackals.set_world_poses(root_pos[:], new_jackal_rot)

        root_pos = self.initial_left_pos.clone()
        root_pos[:, 0] += left_door
        self._left_door.set_world_poses(root_pos[:], self.initial_root_rot[:].clone())

        root_pos = self.initial_right_pos.clone()
        root_pos[:, 0] += right_door
        self._right_door.set_world_poses(root_pos[:], self.initial_root_rot[:].clone())


    # def post_physics_step(self):
    #     self.progress_buf[:] += 1

    #     # self.refresh_dof_state_tensors()
    #     # self.refresh_body_state_tensors()

    #     # self.update_selected_object()

    #     # self.common_step_counter += 1
    #     # if self.common_step_counter % self.push_interval == 0:
    #     #     self.push_robots()
        
    #     # prepare quantities
    #     # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
    #     # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
    #     # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
    #     # forward = quat_apply(self.base_quat, self.forward_vec)
    #     # heading = torch.atan2(forward[:, 1], forward[:, 0])
    #     # self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    #     # self.check_termination()

    #     # if self._selected_id is not None:
    #     #     self.commands[self._selected_id, :] = torch.tensor(self._current_command, device=self.device)
    #     #     self.timeout_buf[self._selected_id] = 0
    #     #     self.reset_buf[self._selected_id] = 0

    #     # self.get_states()

    #     env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    #     if len(env_ids) > 0:
    #         self.reset_idx(env_ids)

    #     self.get_observations()
    #     # if self.add_noise:
    #     #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    #     # self.last_actions[:] = self.actions[:]
    #     # self.last_dof_vel[:] = self.dof_vel[:]

    #     return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
