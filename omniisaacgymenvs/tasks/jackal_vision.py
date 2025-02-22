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
import omni.replicator.core as rep
from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener
from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter
from gym import spaces
# import matplotlib.pyplot as plt

from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdPhysics, UsdLux
import omni

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from pxr import PhysicsSchemaTools, PhysxSchema

import numpy as np
import torch
import math

"""
To-Do
1. Merge changes to environment from Jackal task
2. Add more lighting so it is consistent throughout the environment
3.
"""


class JackalVisionTask(RLTask):
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

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_velocity = 20.0
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self._num_observations = 49152
        self._num_actions = 2

        self._stacked_images = self._task_cfg["env"]["stackedImages"]
        self._img_width = self._task_cfg["env"]["imgWidth"]
        self._img_height = self._task_cfg["env"]["imgHeight"]
        self._img_chs = self._task_cfg["env"]["imgChannels"]

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self._img_width, self._img_height, self._img_chs*self._stacked_images), dtype=np.float32)
        

        RLTask.__init__(self, name, env)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.target_position = torch.tensor([0.5, 2.0, 0], device=self.device)

        self.obs_buf = torch.zeros((self.num_envs, self._img_width, self._img_height, self._img_chs*self._stacked_images), device=self.device, dtype=torch.float)
        self._past_oberservations = torch.zeros((self.num_envs, self._img_width, self._img_height, self._img_chs*(self._stacked_images-1)), device=self.device, dtype=torch.float)

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
        scene.add_ground_plane(size=800.0, color=torch.tensor([0.01,0.01,0.01]))

        self.root_pos, self.root_rot = self._jackals.get_world_poses(clone=False)
        # self.dof_pos = self._jackals.get_joint_positions(clone=False)
        # self.dof_vel = self._jackals.get_joint_velocities(clone=False)
        self.left_root_pos, self.left_root_rot = self._left_door.get_world_poses(clone=False)
        self.right_root_pos, self.right_root_rot = self._right_door.get_world_poses(clone=False)

        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()
        self.initial_left_pos, self.initial_left_rot = self.left_root_pos.clone(), self.left_root_rot.clone()
        self.initial_right_pos, self.initial_right_rot = self.right_root_pos.clone(), self.right_root_rot.clone()

        # prim_path="/World/defaultDistantLight", intensity=5000
        # stage = get_current_stage()
        # light = UsdLux.DistantLight.Define(stage, prim_path)
        # light.GetPrim().GetAttribute("intensity").Set(intensity)


        render_products = []
        for i in range(self.num_envs):
            camera= rep.create.camera(position=(self.root_pos[i,:].cpu().numpy() + np.array([0.0,-3.0,6.0])), rotation=(0.0,-50.0,-90))
            rp = rep.create.render_product(camera,resolution=(self._img_width,self._img_height))
            render_products.append(rp)

        self.listener = PytorchListener()
        rep.WriterRegistry.register(PytorchWriter)
        self.writer = rep.WriterRegistry.get("PytorchWriter")
        self.writer.initialize(listener=self.listener,device=self.device)
        self.writer.attach(render_products)


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
        
        # root_positions = self.root_pos
        # self.obs_buf[..., 0:3] = root_positions
        # self.obs_buf[..., 3:7] = self.root_rot
        # self.obs_buf[..., 7] = self.root_right_pos[:,0]
        # self.obs_buf[..., 8] = self.root_left_pos[:,0]

        self.obs_buf = torch.zeros((self.num_envs, self._img_width, self._img_height, self._img_chs*self._stacked_images), device=self.device, dtype=torch.float)


        current_obs =self.listener.get_rgb_data().permute(0, 2, 3, 1)

        self.obs_buf[:,:,:,0:self._img_chs*(self._stacked_images-1)] = self._past_oberservations
        self.obs_buf[:,:,:,self._img_chs*(self._stacked_images-1):] = current_obs


        self._past_oberservations = self.obs_buf[:,:,:,self._img_chs*1:]


        # self.imgplot = plt.imshow(self.obs_buf.cpu().numpy()[0,:,:,:])
        # plt.savefig("mygraph.png")

        # self.obs_buf = self.obs_buf


        # .view(self.num_envs, 3, 128,128)
        # .flatten(start_dim=1)

            

        observations = self.obs_buf

        return observations

    def pre_physics_step(self, actions) -> None:

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)
        torch.where(actions > 1.0, (actions-1)*-1, actions)
        # actions = actions - 1
        # actions = actions.repeat(1, 2)
        # Add action conditional later
        velocity = torch.zeros((self._jackals.count, self._jackals.num_dof), dtype=torch.float32, device=self._device)

        velocity[:,0],velocity[:,1] = self.wheel_velocities(actions[:,0]*2,actions[:,1]*30)
        velocity[:,2],velocity[:,3] = self.wheel_velocities(actions[:,0]*2,actions[:,1]*30)
        


        # forces = torch.zeros((self._jackals.count, self._jackals.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        indices = torch.arange(self._jackals.count, dtype=torch.int32, device=self._device)
        self._jackals.set_joint_velocity_targets(velocity, indices=indices)

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

        # self._past_oberservations[env_ids,:,:,:] = torch.zeros((num_resets, self._img_width, self._img_height, self._img_chs*(self._stacked_images-1)), device=self.device, dtype=torch.float)
        obs_array = torch.zeros((self.num_envs, self._img_width, self._img_height, self._img_chs*(self._stacked_images-1)), device=self.device, dtype=torch.float)
        self._past_oberservations[env_ids,:,:,:] = obs_array[env_ids,:,:,:]


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
        jackal_pos = self.root_pos

        # cart_pos = self.obs_buf[:, 0]
        # cart_vel = self.obs_buf[:, 1]
        # pole_angle = self.obs_buf[:, 2]
        # pole_vel = self.obs_buf[:, 3]

        dist = ((jackal_pos-self.target_position)**2).sum(axis=1)
        reward1 = torch.exp(dist/10)*-1
        reward2 = torch.where(torch.abs(dist) < 0.5, 100, 0)
        reward = reward1+reward2


        # reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        jackal_pos = self.root_pos


        dist = ((jackal_pos-self.target_position)**2).sum(axis=1)

        # pole_pos = self.obs_buf[:, 2]

        resets = torch.where(torch.abs(dist) < 0.5, 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), resets)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets


    def wheel_velocities(self, linear_velocity, angular_volicity):

        return (linear_velocity - (0.38/2)*angular_volicity)/0.1, (linear_velocity + (0.38/2)*angular_volicity)/0.1


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
