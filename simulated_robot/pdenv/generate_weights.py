import os

import gym
import numpy as np
import matplotlib.pyplot as plt

import gym_panda
import argparse
from itertools import count

import scipy.optimize

import torch
import pickle
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG,PPO2, TRPO


def generate_weight(DISMODEL_PATH,NORMALMODEL_PATH, FULL_DIS_DEMONS, FULL_NORMAL_DEMONS, DIS_DEMONS, FULL_DEMONS, SAVE_PATH):
    env = gym.make("feasibilitypanda-v0")
    seed = 22
    env.seed(seed)
    torch.manual_seed(seed)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    policy_indices = list(range(num_inputs))
    dispolicy_net = Policy(len(policy_indices), num_actions)
    normalpolicy_net = Policy(len(policy_indices), num_actions)
    dispolicy_net.load_state_dict(torch.load(DISMODEL_PATH))
    normalpolicy_net.load_state_dict(torch.load(NORMALMODEL_PATH))

    # get data
    full_disabled =  pickle.load(open(FULL_DIS_DEMONS, 'rb'))
    full_normal =  pickle.load(open(FULL_NORMAL_DEMONS, 'rb'))
    disabled =  pickle.load(open(DIS_DEMONS, 'rb'))
    normal =  pickle.load(open(FULL_DEMONS, 'rb'))
    discount = 0.9

    weights = np.zeros(disabled.shape[0] + normal.shape[0])  #shape is num of traj
    for j in range(disabled.shape[0]): 
        print("disabled",j)
        trajweight = np.zeros(disabled[j].shape)  #shape is length of current traj
        state = env.reset(j,"dis")
        i = 0
        done = False
        while not done:
            state = torch.from_numpy(state).unsqueeze(0)
            action_mean, _, action_std = dispolicy_net(Variable(state).float())
            state, rewards, done, info = env.step(action_mean)
            dis = info['dis']  #calculate the exp distance
            trajweight[i] = dis*discount**(i)
            i = i + 1
        weights[j] = -np.sum(trajweight) #update weight traj by traj
    for j in range(normal.shape[0]):   #normal now is 48
        print("normal",j)
        trajweight = np.zeros(normal[j].shape)  #shape is length of current traj
        state = env.reset(j,"normal")
        i = 0
        done = False
        while not done:
            state = torch.from_numpy(state).unsqueeze(0)
            action_mean, _, action_std = normalpolicy_net(Variable(state).float())
            state, rewards, done, info = env.step(action_mean)
            dis = info['dis']  #calculate the exp distance
            trajweight[i] = dis*discount**(i)
            i = i + 1
        weights[j+5] = -np.sum(trajweight) #update weight traj by traj
    
    
    print("average:",np.mean(weights))
    # rescale weights
    maxweight = np.max(weights)
    low = weights.shape[0]/10
    high = weights.shape[0]*9/10
    #pdb.set_trace()
    alpha = np.abs(np.mean(weights[int(low):int(high)])-maxweight)/10
    weights = np.exp((weights -maxweight) /alpha)
    print(weights)
    data = np.concatenate((disabled,normal),axis= 0)
    pickle.dump(weights, open(SAVE_PATH, 'wb'))
            

if __name__ == "__main__":
    generate_weight('..\\data\\dis5_3model',
                    '..\\data\\normal48_3model',  
                    '..\\data\\dis5.pkl',
                    '..\\data\\normal48.pkl',
                    '..\\data\\dis5ee.pkl',
                    '..\\data\\normal48ee.pkl',
                    '..\\data\\weights_3.pkl')
