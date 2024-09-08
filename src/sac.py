import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy, Action_net
import re
bus_num = [0, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1]
client_location = [2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 10, 11, 11, 12, 13, 14]


def find_neighbor(id):
    nb = []
    for i in range(len(client_location)):
        if client_location[i] == id:
            nb.append(i)
    return nb


class SAC(object):
    def __init__(self, num_inputs, action_space, args, id):
        # parameter for updating
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.id = int(id)
        self.nb = find_neighbor(self.id)
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        # self.g_net = Global_net(args.num_agent).to(device=self.device)
        # self.critic = Q_net.to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.clr)
        self.action_net = Action_net(action_space.shape[0] * len(self.nb))
        self.action_optim = Adam(self.action_net.parameters(), lr=args.clr)
        # self.critic_optim = SGD(self.critic.parameters(), lr=args.clr, momentum=0.9)
        # self.critic_scheduler = MultiStepLR(self.critic_optim, milestones=[2000, 6000], gamma=0.1)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size). \
            to(self.device)
        # self.critic_target = Q_target.to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                # torch.prod:output product of all elements in the action space tensor
                # shape return the dimension of action space
                # Target Entropy = −dim(A)
                self.target_entropy = -torch.prod(torch.Tensor(torch.tensor([]).shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                # log_alpha is the updated parametber in SAC
                self.alpha_optim = Adam([self.log_alpha], lr=args.alphalr)

                # self.alpha_optim = SGD([self.log_alpha], lr=args.alphalr, momentum=0.9)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size,
                                         action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.plr)

            # self.policy_optim = SGD(self.policy.parameters(), lr=args.plr, momentum=0.9)
            # self.policy_scheduler = MultiStepLR(self.policy_optim, milestones=[2000, 6000], gamma=0.1)

        else:  # DeterministicPolicy method
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.plr)

    def select_action(self, state, evaluate=False):
        # FloatTensor:32-bit floating point
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates, id):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, other_action_batch, other_action_next_batch\
            = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        other_action_batch = torch.FloatTensor(other_action_batch).to(self.device)
        other_action_next_batch = torch.FloatTensor(other_action_next_batch).to(self.device)
        # print(other_action_batch.shape)
        # disabled gradient calculation, calculate the loss of q network
        with torch.no_grad():
            nb_action_n = []
            nb_action = []
            next_state_action, next_state_log_pi, _ = self.policy.sample(torch.cat((next_state_batch[:, 0:48], state_batch[:, 48 + id * 5: 48 + (id + 1) * 5]), 1))
            # for nei in self.nb:
            #     nb_action_n.append(other_action_next_batch[:, nei])
            #     nb_action.append(other_action_batch[:, nei])
            # other_action_n = torch.cat(nb_action_n, dim=1)
            # other_action = torch.cat(nb_action, dim=1)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, torch.mean(other_action_next_batch, dim=1, keepdim=True))
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, self.action_net(other_action_n))
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, other_action_next_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)  # - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch, torch.mean(other_action_batch, dim=1, keepdim=True))  # Two Q-functions to mitigate positive bias in the policy improvement step
        # qf1, qf2 = self.critic(state_batch, action_batch, other_action_batch)
        # qf1, qf2 = self.critic(state_batch, action_batch, self.action_net(other_action_n))
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # g_loss = torch.mean(self.g_net(other_action_batch))
        # qf1_loss += g_loss
        # qf2_loss += g_loss
        qf_loss = qf1_loss + qf2_loss

        # update q network parameters
        self.critic_optim.zero_grad()
        self.action_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        self.action_optim.step()
        # self.critic_scheduler.step()

        pi, log_pi, _ = self.policy.sample(torch.cat((state_batch[:, 0:48], state_batch[:, 48 + id * 5: 48 + (id + 1) * 5]), 1))

        # qf1_pi, qf2_pi = self.critic(state_batch, pi, other_action_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi, torch.mean(other_action_batch, dim=1, keepdim=True))
        # qf1_pi, qf2_pi = self.critic(state_batch, pi, self.action_net(other_action))
        # qf1_pi += g_loss
        # qf2_pi += g_loss
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))] SAC
        # policy_loss = -1.0 * (log_pi * min_qf_pi).mean()  # AC
        policy_loss = -1.0 * min_qf_pi.mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        # self.policy_scheduler.step()
        # update par alpha automatically through calculating alpha loss
        # if self.automatic_entropy_tuning:
        #     alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        #
        #     self.alpha_optim.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha_optim.step()
        #
        #     self.alpha = self.log_alpha.exp()
        #     alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        # else:
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        # item()-tuple (key,value)
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

