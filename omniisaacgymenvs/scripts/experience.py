#!/usr/bin/env python
import numpy as np
import torch
import ros, time, pickle, math, copy, random

class Experience(object):

    def __init__(self, prior_alpha = 0.0, prior_beta=0.0, length_scale=1.0, num_env = 1):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.length_scale = length_scale
        self.episode_list = []
        self.replay_buffer_episodes = []
        self.successful_states = []
        self.unsuccessful_states =[]

        self.num_env = num_env

        for i in range(self.num_env):
        	self.episode_list.append([])

    def add_step(obs, reward, reset):
    	for index in range(self.num_env):
    		self.episode_list[index].append(obs[index,:])
    		if reward[index] > 0 and reset[index] == 1:
    			self.successful_states.append(self.episode_list[index])
    			self.episode_list.pop(index)
    		elif reset[index] == 1:
    			self.unsuccessful_states.append(self.episode_list[index])
    			self.episode_list.pop(index)

