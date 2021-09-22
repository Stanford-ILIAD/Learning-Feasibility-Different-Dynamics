import argparse
from itertools import count

import gym
import scipy.optimize

import pdb
import torch
from torch.autograd import Variable

from jax_rl.agents import AWACLearner, SACLearner
from jax_rl.datasets import ReplayBuffer
from jax_rl.evaluation import evaluate
from jax_rl.utils import make_env

import numpy as np

import pickle
import random
import copy

import ant
import swimmer
import reacher
import walker
import halfcheetah
import inverted_double_pendulum

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save_path', type=str, default= 'temp', metavar='N',
                    help='path to save demonstrations on')
parser.add_argument('--xml', type=str, default= None, metavar='N',
                    help='For diffent dynamics')
parser.add_argument('--demo_files', nargs='+')
parser.add_argument('--test_demo_files', nargs='+')
parser.add_argument('--ratio', type=float, nargs='+')
parser.add_argument('--eval-interval', type=int, default=1000)
parser.add_argument('--restore_model', default=None)
parser.add_argument('--mode')
parser.add_argument('--discount', type=float, default=0.9)
parser.add_argument('--discount_train', action='store_true')
parser.add_argument('--fixed_train', action='store_true')
parser.add_argument('--algo', default='sac', help='the algorithm of RL')
parser.add_argument('--max_steps', type=int, default=int(1e6), help='the maximum number of steps')
parser.add_argument('--start_training', type=int, default=int(1e4), help='Number of training steps to start training.')
args = parser.parse_args()

def load_demos(demo_files, ratio):
    all_demos = []
    all_init_obs = []
    for demo_file in demo_files:
        raw_demos = pickle.load(open(demo_file, 'rb'))
        use_num = int(len(raw_demos['obs'])*ratio)
        all_demos = all_demos + raw_demos['obs'][:use_num]
        if 'init_obs' in raw_demos:
            all_init_obs = all_init_obs + raw_demos['init_obs'][:use_num]
    return all_demos, all_init_obs

def load_pairs(demo_files, ratio):
    all_pairs = []
    for demo_file in demo_files:
        raw_demos = pickle.load(open(demo_file, 'rb'))
        for i in range(int(len(raw_demos['obs'])*ratio)):
            obs = np.array(raw_demos['obs'][i])
            next_obs = np.array(raw_demos['next_obs'][i])
            all_pairs.append(np.reshape(np.concatenate([obs, next_obs], axis=1), (obs.shape[0], 2, -1)))
    return np.concatenate(all_pairs, axis=0)

#demos = load_demos(args.demo_file)
if args.mode == 'pair':
    demos = [load_pairs(args.demo_files[i:i+1], args.ratio[i]) for i in range(len(args.test_demo_files))]
elif args.mode == 'traj':
    demos = []
    init_obs = []
    for i in range(len(args.test_demo_files)):
        demos_single, init_obs_single = load_demos(args.demo_files[i:i+1], args.ratio[i])
        demos.append(demos_single)
        init_obs.append(init_obs_single)
test_demos = []
test_init_obs = []
for i in range(len(args.test_demo_files)):
    demos_single, init_obs_single = load_demos(args.test_demo_files[i:i+1], args.ratio[i])
    test_demos.append(demos_single)
    test_init_obs.append(init_obs_single)

if 'Ant' in args.env_name or 'Swimmer' in args.env_name or 'Walker' in args.env_name or 'HalfCheetah' in args.env_name:
    env_list = [gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos[i]) for i in range(len(args.demo_files))]
    test_env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos[0][0:3])
elif 'InvertedDoublePendulum' in args.env_name:
    env_list = [gym.make(args.env_name, xml_file=args.xml, demos=demos[i], init_obs=init_obs[i]) for i in range(len(args.demo_files))]
    test_env = gym.make(args.env_name, xml_file=args.xml, demos=demos[0][0:3], init_obs=init_obs[0][0:3])
else:
    env_list = [gym.make(args.env_name, xml_file=args.xml, demos=demos[i]) for i in range(len(args.demo_files))]
    test_env = gym.make(args.env_name, xml_file=args.xml, demos=demos[0][0:3])

num_inputs = env_list[0].observation_space.shape[0]
num_actions = env_list[0].action_space.shape[0]

for i in range(len(env_list)):
    env_list[i].seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)


agents = []
if args.restore_model is not None:
    model_dict = torch.load(args.restore_model)
    for i in range(len(model_dict)):
        agents[i].append(model_dict['policy_'+str(i)])
else:
    if args.algo == 'sac':
        from configs.sac_default import get_config
        kwargs = dict(get_config())
        algo = kwargs.pop('algo')
        replay_buffer_size = kwargs.pop('replay_buffer_size')
        for i in range(len(demos)):
            agents.append(SACLearner(args.seed,
                          np.concatenate([env_list[0].observation_space.sample()[np.newaxis], env_list[0].observation_space.sample()[np.newaxis]], axis=1),
                          env_list[0].action_space.sample()[np.newaxis], **kwargs))
    elif args.algo == 'awac':
        from configs.awac_default import get_config
        kwargs = dict(get_config())
        algo = kwargs.pop('algo')
        replay_buffer_size = kwargs.pop('replay_buffer_size')
        for i in range(len(demos)):
            agents.append(AWACLearner(args.seed,
                        np.concatenate([env_list[0].observation_space.sample()[np.newaxis], env_list[0].observation_space.sample()[np.newaxis]], axis=1),
                        env_list[0].action_space.sample()[np.newaxis], **kwargs))
    else:
        raise NotImplementedError()

replay_buffer_list = []
best_reward_list = []
save_model_dict = {}
for i in range(len(demos)):
    action_dim = env_list[i].action_space.shape[0]
    replay_buffer_list.append(ReplayBuffer(gym.spaces.Box(np.concatenate([env_list[i].observation_space.low, env_list[i].observation_space.low], axis=0),
                                                          np.concatenate([env_list[i].observation_space.high, env_list[i].observation_space.high], axis=0),
                                                          [num_inputs*2]), 
                                           action_dim, replay_buffer_size or args.max_steps))
    best_reward_list.append(-10000000)
    save_model_dict['policy_'+str(i)] = None

def evaluate(test_demos, agents, test_env, best_reward_list, num_episode, test_init_obs):
    for test_demo_id in range(len(test_demos)):
        all_reward = []
        for ii in range(num_episode):
            test_env.reset()
            if len(test_init_obs[test_demo_id]) > 0:
                state = test_env.set_initial_state(test_demos[test_demo_id][ii], test_init_obs[test_demo_id][ii])
            else:
                state = test_env.set_initial_state(test_demos[test_demo_id][ii])
            state0 = state
            done = False
            test_reward = 0
            step_id = 0
            while not done:
                action = agents[test_demo_id].sample_actions(np.concatenate([state,state0], axis=0), temperature=0.0)
                next_state, reward, done, _ = test_env.step(action)
                if args.mode == 'pair':
                    test_reward += reward
                    state = test_demos[test_demo_id][ii][test_env.step_]
                    test_env.set_observation(state)
                elif args.mode == 'traj':
                    test_reward += reward * (args.discount**step_id)
                    state = next_state
                step_id += 1
            #test_reward /= len(test_demos[test_demo_id][ii])
            all_reward.append(test_reward)       
        print('reward', test_demo_id, ' ', np.mean(all_reward))
        if best_reward_list[test_demo_id] < np.mean(all_reward):
            best_reward_list[test_demo_id] = np.mean(all_reward)
            save_model_dict['policy_'+str(test_demo_id)] = copy.deepcopy(agents[test_demo_id])
        print('best reward', test_demo_id, ' ', best_reward_list[test_demo_id])
    torch.save(save_model_dict, args.save_path + 'seed_{}model'.format(args.seed))
    torch.save({'policy_'+str(i):agents[i] for i in range(len(test_demos))}, args.save_path+'seed{}instantmodel'.format(args.seed))

observation_list = []
done_list = []
step_id_list = []

state0_list = []
for j in range(len(demos)):
    observation, done = env_list[j].reset(), False
    observation_list.append(observation)
    done_list.append(done)
    step_id_list.append(0)
    state0_list.append(observation)
for i in range(1, args.max_steps + 1):
    for j in range(len(demos)):
        if i < args.start_training:
            action = env_list[j].action_space.sample()
        else:
            action = agents[j].sample_actions(np.concatenate([observation_list[j], state0_list[j]], axis=0))
        next_observation, reward, done, info = env_list[j].step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        if args.discount_train:
            replay_buffer_list[j].insert(np.concatenate([observation_list[j], state0_list[j]], axis=0), action, reward * (args.discount**step_id_list[j]), mask, np.concatenate([next_observation, state0_list[j]], axis=0))
        else:
            replay_buffer_list[j].insert(np.concatenate([observation_list[j], state0_list[j]], axis=0), action, reward, mask, np.concatenate([next_observation, state0_list[j]], axis=0))
        step_id_list[j] += 1

        observation_list[j] = next_observation
        done_list[j] = done

        if done:
            observation_list[j], done_list[j] = env_list[j].reset(), False
            step_id_list[j] = 0
            state0_list[j] = observation_list[j]

        if i >= args.start_training:
            batch = replay_buffer_list[j].sample(args.batch_size)
            update_info = agents[j].update(batch)

    if i % args.eval_interval == 0 and i >= args.start_training:
        evaluate(test_demos, agents, test_env, best_reward_list, 10, test_init_obs)

