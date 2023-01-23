import numpy as np
import torch
import time, pickle, math, copy, random

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

    def add_step(self, obs, reward, reset):
    	for index in range(self.num_env):
    		self.episode_list[index].append(obs[index,:])
    		if reward[index] > 0 and reset[index] == 1:
    			self.successful_states.append(self.episode_list[index])
    			self.episode_list[index] = []

    		elif reset[index] == 1:
    			self.unsuccessful_states.append(self.episode_list[index])
    			self.episode_list[index] = []

    def get_state_value(self, state):
        alpha = self.prior_alpha
        beta = self.prior_beta
        length_scale = self.length_scale

        for episode in self.successful_states:
        	for old_state in episode:

        		state_delta = old_state - state

        		weight = np.exp(-1.0*(np.linalg.norm(state_delta/length_scale))**2)

        		alpha = alpha+weight

        for episode in self.unsuccessful_states:
        	for old_state in episode:
        		state_delta = old_state - state

        		weight = np.exp(-1.0*(np.linalg.norm(state_delta/length_scale))**2)

        		beta = beta+weight

        value = alpha/(alpha + beta)
        variance = alpha*beta/((alpha+beta)**2*(alpha+beta+1.0))
        sigma = np.sqrt(variance)
        return value, sigma, alpha, beta

