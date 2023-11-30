#!/usr/bin/env python
import numpy as np
import torch
import csv

class Dataset(object):
    #assume there is only a single environment

    def __init__(self, dataset_size=0, start_states=None):
        self.labels = []
        self.start_states = start_states
        self.frame_count = []
        self.dataset_size = dataset_size

        self.episode_list = []
        self.current_env_index = 0

        self.replay_buffer_episodes = []
        self.successful_states = []
        self.unsuccessful_states =[]

        for i in range(dataset_size):
            self.episode_list.append([])

    def add_frame_count(self, frame_count):
        self.frame_count.append(frame_count)
    
    def add_label(self, label):
        self.labels.append(label)
        self.current_env_index +=1

    def add_step(self, obs, reward, reset):
    		self.episode_list[self.current_env_index].append(obs[0,:])
    		if reward[0] > 0 and reset[0] == 1:
    			self.successful_states.append(self.episode_list[self.current_env_index])
    		elif reset[0] == 1:
    			self.unsuccessful_states.append(self.episode_list[self.current_env_index])

    def get_success_rate(self):
        return len(self.successful_states)/(len(self.successful_states)+len(self.unsuccessful_states))

    def save(self, dataset_file_name, metadata_file_name):
        # save files as csv format
        # save each obs in the csv with the corresponding label from self.labels
        with open(dataset_file_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for i, episode in enumerate(self.episode_list):
                for obs in episode:
                    writer.writerow([*obs, self.labels[i]])

        # save the start states in a csv with the correct label and frame count
        with open(metadata_file_name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(self.dataset_size):
                writer.writerow([*self.start_states[i], self.labels[i], self.frame_count[i]])
        
