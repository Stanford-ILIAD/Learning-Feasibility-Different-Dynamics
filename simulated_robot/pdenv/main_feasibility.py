import argparse
from itertools import count

import gym
import gym_panda
import scipy.optimize

import pdb
import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
#from gen_dem import demo
import pickle
import random

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
parser.add_argument('--discount', type=float, default=0.9)
parser.add_argument('--discount_train', action='store_true')
parser.add_argument('--mode',help='dis or normal to create rl model for disabled panda or normal panda')
parser.add_argument('--disdata',help='disabled panda data')
parser.add_argument('--normaldata',help='normal panda data')

args = parser.parse_args()

if args.mode == 'normal':
    data =  pickle.load(open(args.normaldata, 'rb')) #48 traj
    test_demos = data
    demos = data
    num=48
else:
    data =  pickle.load(open(args.disdata, 'rb')) #5 traj
    disabledpdata = data
    test_demos = data
    demos = data
    num=5

test_env = gym.make(args.env_name)
env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
test_env.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

if args.restore_model is not None:
    model_dict = torch.load(args.restore_model)
    policy_net.load_state_dict(model_dict['policy_'+str(i)])

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


def update_params(batch, policy_n, value_n):  #policy net and value net
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

best_reward = -10000000
save_model_dict ={'policy_'+str(i):None for i in range(len(test_demos))} 

for i_episode in count(1):
    if i_episode % args.eval_interval == 1:
        for test_demo_id in range(1):
            all_reward = []
            for ii in range(num):
                test_env.reset()
                jointposition = np.concatenate((test_demos[ii][0][:9],np.array([0.03,0.03])),axis=None)
                test_env.panda._reset_robot(jointposition)  #change into joint position
                state = test_env.panda.state['ee_position']
                done = False
                test_reward = 0
                step_id = 0
                while not done:
                    action = select_action_test(state, policy_net)
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = test_env.step(action)
                    test_reward += reward * (args.discount**step_id)
                    state = next_state
                    step_id += 1
                all_reward.append(test_reward)
            print(all_reward)
            print('reward', ' ', np.mean(all_reward))
            if best_reward < np.mean(all_reward):
                best_reward = np.mean(all_reward)
                torch.save(policy_net.state_dict(), args.save_path+'model')
            print('best reward: ', best_reward)
        
    memory = Memory()
    
    for i in range(1):
        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        while num_steps < args.batch_size:
            state = env.reset()
            #state = running_state(state)
            state0 = state
            reward_sum = 0
            step_id = 0
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(state, policy_net)
                action = action.data[0].numpy()
                next_state, reward, done, _ = env.step(action)
                
                if args.discount_train:
                    reward_sum += reward * (args.discount**step_id)
                else:
                    reward_sum += reward

                #next_state = running_state(next_state)

                mask = 1
                if done:
                    mask = 0

                if args.discount_train:
                    memory.push(state, np.array([action]), mask, next_state, reward * (args.discount**step_id))
                else:
                    memory.push(state, np.array([action]), mask, next_state, reward)

                if args.render:
                    env.render()
                if done:
                    break

                state = next_state
                step_id += 1
            num_steps += t+1
            num_episodes += 1
            reward_batch += reward_sum
        reward_batch /= num_episodes
        batch = memory.sample()
        update_params(batch, policy_net, value_net)