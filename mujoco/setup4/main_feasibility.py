import argparse
from itertools import count

import gym
import scipy.optimize

import pdb
import torch
from models.old_models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
import pickle
import random

import swimmer
import walker
import halfcheetah

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
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
parser.add_argument('--eval-interval', type=int, default=10)
parser.add_argument('--restore_model', default=None)
parser.add_argument('--mode')
parser.add_argument('--discount', type=float, default=0.9)
parser.add_argument('--discount_train', action='store_true')
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

env_list = [gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos[i]) for i in range(len(args.demo_files))]
test_env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False, demos=demos[0][0:3])

num_inputs = env_list[0].observation_space.shape[0]
num_actions = env_list[0].action_space.shape[0]

for i in range(len(env_list)):
    env_list[i].seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

policy_nets = [Policy(num_inputs*2, num_actions) for i in range(len(args.demo_files))]
value_nets = [Value(num_inputs*2) for i in range(len(args.demo_files))]

if args.restore_model is not None:
    model_dict = torch.load(args.restore_model)
    for i in range(len(policy_nets)):
        policy_nets[i].load_state_dict(model_dict['policy_'+str(i)])

def select_action_test(state, policy):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action_mean

def select_action(state, policy):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action


def update_params(batch, policy_n, value_n):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_n(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_n, torch.Tensor(flat_params))
        for param in value_n.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_n(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_n.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_n).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_n).double().numpy(), maxiter=25)
    set_flat_params_to(value_n, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_n(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_n(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_n(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_n(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_n, get_loss, get_kl, args.max_kl, args.damping)


best_reward_list = []
for i in range(len(test_demos)):
    best_reward_list.append(-10000000)
save_model_dict ={'policy_'+str(i):None for i in range(len(test_demos))} 

for i_episode in count(1):
    if i_episode % args.eval_interval == 1:
        for test_demo_id in range(len(test_demos)):
            all_reward = []
            for ii in range(10):
                test_env.reset()
                if 'InvertedDoublePendulum' in str(type(test_env.env)):
                    state = test_env.set_initial_state(test_demos[test_demo_id][ii], test_init_obs[test_demo_id][ii])
                else:
                    state = test_env.set_initial_state(test_demos[test_demo_id][ii])
                state0 = state
                done = False
                test_reward = 0
                step_id = 0
                while not done:
                    action = select_action_test(np.concatenate([state, state0], axis=0), policy_nets[test_demo_id])
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = test_env.step(action)
                    if args.mode == 'pair':
                        test_reward += reward
                        state = test_demos[test_demo_id][ii][test_env.step_]
                        test_env.set_observation(state)
                    elif args.mode == 'traj':
                        test_reward += reward * (args.discount**step_id)
                        state = next_state
                    step_id += 1
                all_reward.append(test_reward)
            print(all_reward)
            print('reward', test_demo_id, ' ', np.mean(all_reward))
            if best_reward_list[test_demo_id] < np.mean(all_reward):
                best_reward_list[test_demo_id] = np.mean(all_reward)
                save_model_dict['policy_'+str(test_demo_id)] = policy_nets[test_demo_id].state_dict()
            print('best reward', test_demo_id, ' ', best_reward_list[test_demo_id])
        torch.save(save_model_dict, args.save_path + 'seed_{}'.format(args.seed))
        
    memory_list = [Memory() for i in range(len(demos))]
    
    for i in range(len(demos)):
        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        while num_steps < args.batch_size:
            state = env_list[i].reset()
            state0 = state
            reward_sum = 0
            step_id = 0
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(np.concatenate([state, state0], axis=0), policy_nets[i])
                action = action.data[0].numpy()
                next_state, reward, done, _ = env_list[i].step(action)
                
                if args.discount_train:
                    reward_sum += reward * (args.discount**step_id)
                else:
                    reward_sum += reward


                mask = 1
                if done:
                    mask = 0

                if args.discount_train:
                    memory_list[i].push(np.concatenate([state, state0], axis=0), np.array([action]), mask, next_state, reward * (args.discount**step_id))
                else:
                    memory_list[i].push(np.concatenate([state, state0], axis=0), np.array([action]), mask, next_state, reward)

                if args.render:
                    env_list[i].render()
                if done:
                    break

                state = next_state
                step_id += 1
            num_steps += t+1
            num_episodes += 1
            reward_batch += reward_sum
        reward_batch /= num_episodes
        batch = memory_list[i].sample()
        update_params(batch, policy_nets[i], value_nets[i])
