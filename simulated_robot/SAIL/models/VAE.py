import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch.distributions import Normal
from models.ppo_models import weights_init_
import pdb

MAX_LOG_STD = 0.5
MIN_LOG_STD = -20

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_size=128):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_size=128):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, out_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class VAE(torch.nn.Module):
    def __init__(self, state_dim, hidden_size=128, latent_dim=128):
        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(state_dim, latent_dim=latent_dim, hidden_size=self.hidden_size)
        self.decoder = Decoder(latent_dim, state_dim, hidden_size=self.hidden_size)

    def forward(self, state):
        mu, log_sigma = self.encoder(state)
        sigma = torch.exp(log_sigma)
        sample = mu + torch.randn_like(mu)*sigma
        self.z_mean = mu
        self.z_sigma = sigma

        return self.decoder(sample)

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def get_next_states(self, states):
        mu, log_sigma = self.encoder(states)
        return self.decoder(mu)

    def get_loss(self, state, next_state):
        next_pred = self.get_next_states(state)
        return nn.L1Loss()(next_pred,next_state)

    def train(self, input, target, epoch, optimizer, batch_size=128, beta=0.1, use_weight=True):
        #pdb.set_trace()
        idxs = np.arange(input.shape[0])
        np.random.shuffle(idxs)
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))
        #pdb.set_trace()
        for epoch in range(epoch):
            idxs = np.arange(input.shape[0])
            np.random.shuffle(idxs)
            for batch_num in range(num_batch):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                train_in = input[batch_idxs].float()
                #print(train_in[:10])
                '''if use_weight:
                    train_targ = target[batch_idxs, :-1].float()
                    weights = target[batch_idxs, -1].float().unsqueeze(1)'''
                #else:
                train_targ = target[batch_idxs].float()
                #print(train_targ[:10])

                optimizer.zero_grad()
                dec = self.forward(train_in)
                #print("dec",dec[:10])
                #pdb.set_trace()
                '''if use_weight:
                    reconstruct_loss = (weights*(train_targ-dec)**2).mean()'''
                #else:
                reconstruct_loss = nn.L1Loss()(dec, train_targ)
                '''if reconstruct_loss.item()<0.005:
                    print(train_targ[:10])
                    print("dec",dec[:10])'''


                ll = latent_loss(self.z_mean, self.z_sigma)
                loss = reconstruct_loss + beta*ll
                loss.backward()
                optimizer.step()
        val_input = input[idxs]
        val_dec = self.get_next_states(val_input)

        '''if use_weight:
            val_target = target[idxs, :-1]
            val_weight = target[idxs, -1].unsqueeze(1)       
            loss = (val_weight*(val_target-val_dec)**2).mean().item()'''
        #else:
        val_target = target[idxs]
        loss = nn.L1Loss()(val_dec, val_target).item()
        return loss