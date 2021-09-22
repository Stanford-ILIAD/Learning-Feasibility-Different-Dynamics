import argparse
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from utils.utils import generate_pairs, generate_pairs_old, process_expert_traj, generate_tuples, adjust_lr
from agents.soft_bc_agent import SoftBC_agent
from utils.utils import normalize_states, normalize_expert_traj
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split
from models.VAE import VAE
from tensorboardX import SummaryWriter
import datetime
import pdb

def get_args():
    parser = argparse.ArgumentParser(description='SAIL arguments')
    parser.add_argument('--env-name', default="pandaenv-random-v0", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--expert-traj-path', metavar='G', type=str,
                        help='path of the expert trajectories')
    parser.add_argument('--weight-path', metavar='G', type=str,
                        help='path of the expert trajectories weights')
    parser.add_argument('--state-dim',type=int,
                        help='Dim of state space')
    parser.add_argument('--size-per-traj',type=int, default=200,
                        help='length per trajectory')
    parser.add_argument('--output-path', metavar='G', type=str,
                        help='location to save the VAE model')

    # VAE params
    parser.add_argument('--model-lr', type=float, default=3e-4, metavar='G',
                        help='learning rate for forward/inverse/vae')
    parser.add_argument('--optim-batch-size', type=int, default=128, metavar='N')
    parser.add_argument('--beta', type=float, default=0.005, help='beta VAE coefficient')
    parser.add_argument('--iter', type=int, default=200, help='iteration times')
    parser.add_argument('--epoch', type=int, default=2, help='epoch')
    parser.add_argument('--lr-decay-rate', type=int, default=50, help='lr_decay_rate')
    parser.add_argument('--weight', type=bool, default=True, help='use weighted demons to train VAE')
    parser.add_argument('--seed', type=int, default=543, metavar='N',help='random seed (default: 1)')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args


def train_vae(args, dtype=torch.float32):
    torch.set_default_dtype(dtype)
    state_dim = args.state_dim
    output_path = args.output_path
    # generate state pairs
    expert_traj_raw = list(pickle.load(open(args.expert_traj_path, "rb")))
    if args.weight:
        weights = list(pickle.load(open(args.weight_path, "rb")))
        state_pairs = generate_pairs(weights,expert_traj_raw, state_dim, args.size_per_traj, max_step=50, min_step=50) # tune the step size if needed.
    else:
        state_pairs = generate_pairs_old(expert_traj_raw, state_dim, args.size_per_traj, max_step=50, min_step=50) # tune the step size if needed.
    # shuffle and split
    idx = np.arange(state_pairs.shape[0])
    np.random.shuffle(idx)
    state_pairs = state_pairs[idx, :]
    split = (state_pairs.shape[0]*19)//20
    state_tuples = state_pairs[:split, :]
    test_state_tuples = state_pairs[split:, :]
    print(state_tuples.shape)
    print(test_state_tuples.shape)


    goal_model = VAE(state_dim, latent_dim=128)
    optimizer_vae = torch.optim.Adam(goal_model.parameters(), lr=args.model_lr)
    save_path = '{}_softbc_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, \
                                                  args.beta)
    writer=SummaryWriter(log_dir=os.path.join(output_path, 'runs/'+save_path))
    state_tuples = torch.from_numpy(state_pairs).to(dtype)
    s, t = state_tuples[:, :state_dim], state_tuples[:, state_dim:2*state_dim]

    state_tuples_test = torch.from_numpy(test_state_tuples).to(dtype)
    s_test, t_test = state_tuples_test[:, :state_dim], state_tuples_test[:, state_dim:2 * state_dim]
        

    for i in range(1, args.iter + 1):
        loss = goal_model.train(s, t, epoch=args.epoch, optimizer=optimizer_vae, \
                                        batch_size=args.optim_batch_size, beta=args.beta, use_weight=args.weight)
        next_states = goal_model.get_next_states(s_test)
        val_error =nn.L1Loss()(next_states,t_test)
        writer.add_scalar('loss/vae', loss, i)
        writer.add_scalar('valid/vae', val_error, i)
        if i % args.lr_decay_rate == 0:
            adjust_lr(optimizer_vae, 2.)
        torch.save(goal_model.state_dict(), os.path.join(output_path, '{}_{}_vae.pt'.format(args.env_name, str(args.beta))))


if __name__ == "__main__":
    args = get_args()
    train_vae(args)
