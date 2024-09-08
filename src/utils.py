import math
import torch
import numpy as np
import shutil
import torch.autograd as Variable

# 高斯策略函数的构造方法
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def SampleFromNormalDistribution(mu, sigma, size, min, max):
    times = 0;
    result = 0
    while times < size:
        sample = np.random.normal(mu, sigma)
        if (sample >= min) and (sample <= max):
            result = sample
            times += 1
    return result


def GaussianNomalization(data, column):
    ans =data.copy()
    a = ans[:, column]
    u = a.mean()
    std = a.std()
    a = (a - u) / std
    ans[:, column] = a
    return ans


def MaxMinNormalization(data, column):
    ans = data.copy()
    a = ans[:, column]
    minn = a.min()
    maxx = a.max()
    a = (a - minn) / (maxx - minn)
    ans[:, column] = a
    return ans

def DecimalNormalization(data, column):
    ans = data.copy()
    a = ans[:, column]
    a = a / 100.0
    ans[:, column] = a
    return ans


class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X