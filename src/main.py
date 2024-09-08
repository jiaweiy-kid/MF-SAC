import argparse
import sys

import numpy as np
import torch
import pandas as pd
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import random
import matplotlib.pyplot as plt
import math
import os
from utils import *
from env import Env
import datetime

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--alpha', type=float, default=0.8, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')

parser.add_argument('--clr', type=float, default=0.0003, metavar='G',
                    help='critic learning rate (default: 0.0003)')
parser.add_argument('--plr', type=float, default=0.0001, metavar='G',
                    help='policy learning rate (default: 0.0003)')
parser.add_argument('--alphalr', type=float, default=0.0002, metavar='G',
                    help='alpha learning rate (default: 0.2)')

args = parser.parse_args()

random.seed()
np.random.seed()

env = Env('..\\price\\trainPrice.xlsx', 1)
env_1 = Env('..\\price\\trainPrice.xlsx', 1)
state = torch.randn(1, 53)
action = torch.tensor([0.])
next_state = torch.randn(1, 53)

agent = SAC(state.shape[1], action, args)
agent_1 = SAC(state.shape[1], action, args)
writer = SummaryWriter('../run/four')

memory = ReplayMemory(args.replay_size, args.seed)
memory_1 = ReplayMemory(args.replay_size, args.seed)
total_numsteps = 0
updates = 0
episode_times = 3000
episode_r = []
epoch_price = []
epoch_anx = []
cr1_lst = []
cr2_lst = []
policy_lst = []
alpha_lst = []
episode_reward = np.array([0.0], dtype='f8')
anx_reward = np.array([0.0], dtype='f8')
price_reward = np.array([0.0], dtype='f8')

# avg_reward = np.array([0.0], dtype='f8')
# avg_anx = np.array([0.0], dtype='f8')
# avg_price = np.array([0.0], dtype='f8')
steps = 0

for i_episode in range(1, episode_times + 1):
    ##########################  Start to Charge ##########################################
    steps = 0
    episode_steps = 0
    done = False
    random.seed()
    state = env.reset()
    state_1 = env_1.reset()
    episode_reward = np.array([0.0], dtype='f8')
    anx_reward = np.array([0.0], dtype='f8')
    price_reward = np.array([0.0], dtype='f8')
    while not done:
        if args.start_steps > total_numsteps:
            action = np.array([np.random.uniform(-0.2, 0.2)], dtype='f8')
            action_1 = np.array([np.random.uniform(-0.2, 0.2)], dtype='f8')
        else:
            action = agent.select_action(state)
            action_1 = agent_1.select_action(state_1)

        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):  # each training step
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha \
                    = agent.update_parameters(memory, args.batch_size, updates)
                critic_1_loss_1, critic_2_loss_1, policy_loss_1, ent_loss_1, alpha_1 \
                    = agent_1.update_parameters(memory_1, args.batch_size, updates)
                cr1_lst.append((critic_1_loss+critic_1_loss_1)/2)
                cr2_lst.append((critic_1_loss+critic_1_loss_1)/2)
                policy_lst.append((policy_loss+policy_loss_1)/2)
                alpha_lst.append(alpha)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward_tuple, action, done = env.step(action)
        next_state_1, reward_tuple_1, action_1, done = env_1.step(action_1)
        n = next_state[:48]
        n_1 = next_state_1[:48]
        n_mean = (n + n_1)/2
        next_state[:48] = n_mean
        next_state_1[:48] = n_mean
        reward = reward_tuple[0]
        reward_1 = reward_tuple_1[0]
        anx = reward_tuple[1]
        price = reward_tuple[2]
        episode_reward += (reward_tuple[0] + reward_tuple_1[0])/2
        anx_reward += (reward_tuple[1] + reward_tuple_1[1])/2
        price_reward += (reward_tuple[2] + reward_tuple_1[2])/2

        memory.push(state, action / 0.2, reward, next_state, float(not done))
        memory_1.push(state_1, action / 0.2, reward, next_state_1, float(not done))
        # memory.push(state, action / 0.2, reward, next_state, 1.0)
        # memory_1.push(state_1, action / 0.2, reward, next_state_1, 1.0)
        state = next_state
        state_1 = next_state_1
        total_numsteps += 1
        steps += 1

    print("Episode:", i_episode, "episode_reward:", episode_reward, "price reward:", price_reward, "anx_reward:",
          anx_reward)
    writer.add_scalar('reward/sum_reward', episode_reward, i_episode)
    writer.add_scalar('reward/price_reward', price_reward, i_episode)
    writer.add_scalar('reward/anx_reward', anx_reward, i_episode)

    episode_r.append(episode_reward)
    epoch_price.append(price_reward)
    epoch_anx.append(anx_reward)

    # avg_reward += episode_reward
    # avg_price += price_reward
    # avg_anx += anx_reward

    # if i_episode % 10 == 0:
    #     avg_reward = avg_reward / 10
    #     avg_price = avg_price / 10
    #     avg_anx = avg_anx / 10
    #     writer.add_scalar('avg_reward/sum_reward', avg_reward / 10, i_episode)
    #     writer.add_scalar('avg_reward/price_reward', avg_price / 10, i_episode)
    #     writer.add_scalar('avg_reward/anx_reward', avg_anx / 10, i_episode)
    #     episode_r.append(avg_reward)
    #     epoch_price.append(avg_price)
    #     epoch_anx.append(avg_anx)
    #     avg_reward = np.array([0.0], dtype='f8')
    #     avg_anx = np.array([0.0], dtype='f8')
    #     avg_price = np.array([0.0], dtype='f8')

    # if args.eval and i_episode % 10 == 0:
    #     avg_reward = np.array([0.0], dtype='f8')
    #     avg_anx = np.array([0.0], dtype='f8')
    #     avg_price = np.array([0.0], dtype='f8')
    #     eval_episodes = 10
    #     for _ in range(eval_episodes):
    #         eval_state = env.reset()
    #         done = False
    #         while not done:
    #             action = agent.select_action(eval_state, evaluate=True)
    #
    #             next_state, reward_tuple, done = env.step(action)
    #             avg_reward += reward_tuple[0]
    #             avg_anx += reward_tuple[1]
    #             avg_price += reward_tuple[2]
    #
    #             eval_state = next_state
    #     avg_reward /= eval_episodes
    #     avg_anx /= eval_episodes
    #     avg_price /= eval_episodes
    #     writer.add_scalar('eval/sum_reward', avg_reward, i_episode)
    #     writer.add_scalar('eval/price_reward', avg_price, i_episode)
    #     writer.add_scalar('eval/anx_reward', avg_anx, i_episode)
    #     episode_r.append(avg_reward)
    #     epoch_price.append(avg_price)
    #     epoch_anx.append(avg_anx)
    episode_steps += 1

torch.save(agent.policy.state_dict(), "..\\run\\four\\policy.pb")
torch.save(agent.critic.state_dict(), "..\\run\\four\\critic.pb")
# fig, rplt = plt.subplots(3)
# rplt[0].plot(range(len(episode_r)), np.array(episode_r), 'r')
# rplt[0].set(xlabel='Training episodes', ylabel='Episode reward')
# rplt[1].plot(range(len(episode_r)), np.array(epoch_price))
# rplt[1].set(xlabel='Training episodes', ylabel='Price reward')
# rplt[2].plot(range(len(episode_r)), np.array(epoch_anx))
# rplt[2].set(xlabel='Training episodes', ylabel='Anxiety reward')
# fig.savefig('..\\run\\four\\pic3.png')

fig, rplt0 = plt.subplots()
# rplt0.set_ylim(-200,0)
rplt0.plot(range(0, episode_times), np.array(episode_r), 'r')
rplt0.set(xlabel='Training episodes', ylabel='Episode reward')
fig.savefig('..\\run\\four\\pic4.png')
fig, rplt1 = plt.subplots()
# rplt1.set_ylim(-100,50)
rplt1.plot(range(0, episode_times), np.array(epoch_price))
rplt1.set(xlabel='Training episodes', ylabel='Price reward')
fig.savefig('..\\run\\four\\pic5.png')
fig, rplt2 = plt.subplots()
# rplt2.set_ylim(-100,0)
rplt2.plot(range(0, episode_times), np.array(epoch_anx))
rplt2.set(xlabel='Training episodes', ylabel='Anxiety reward')
fig.savefig('..\\run\\four\\pic6.png')

# fig, ax = plt.subplots(3)
# ax[0].plot(range(len(cr1_lst)), np.array(cr1_lst))
# ax[1].plot(range(len(cr1_lst)), np.array(cr2_lst))
# ax[2].plot(range(len(cr1_lst)), np.array(policy_lst))
# torch.save(agent.policy.state_dict(), "E:\\master\\V2G based on horizontal FL\\anxious-EV实验记录\\policy 100000.pb")


env.simulation(agent)
