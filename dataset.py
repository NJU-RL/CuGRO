import torch
import torch.nn as nn
import gym
# import d4rl
import numpy as np
import functools
import copy
import os
import torch.nn.functional as F
import tqdm
from scipy.special import softmax
MAX_BZ_SIZE = 1024
soft_Q_update = True

def one_hot_encode_task(task_index, num_tasks):
    encoding = np.zeros(num_tasks)
    encoding[task_index-1] = 1
    return encoding

class Diffusion_buffer(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args=args
        self.normalise_return = args.normalise_return
        self.data = self._load_data(args)
        self.actions = self.data["actions"].astype(np.float32)
        self.states = self.data["states"].astype(np.float32)
        self.rewards = self.data["rewards"].astype(np.float32)
        self.done = self.data["done"]
        one_hot = one_hot_encode_task(args.task, args.num_tasks)
        self.one_hot =np.repeat([one_hot], self.states.shape[0], axis = 0).astype(np.float32)

        self.data_ = {}
        self.data_["actions"] = self.actions
        self.data_["states"] = self.states

        self.returns = self.data["returns"].astype(np.float32)
        self.raw_returns = [self.returns]
        self.raw_values = []
        self.returns_mean = np.mean(self.returns)
        self.returns_std = np.maximum(np.std(self.returns), 0.1)
        print("returns mean {}  std {}".format(self.returns_mean, self.returns_std))
        if self.normalise_return:
            self.returns = (self.returns - self.returns_mean) / self.returns_std
            print("returns normalised at mean {}, std {}".format(self.returns_mean, self.returns_std))
            self.args.returns_mean = self.returns_mean
            self.args.returns_std = self.returns_std
        else:
            print("no normal")
        self.ys = np.concatenate([self.returns, self.actions], axis=-1)
        self.ys = self.ys.astype(np.float32)

        self.len = self.states.shape[0]
        self.data_len = self.ys.shape[0]
        self.fake_len = self.len
        print(self.len, "data loaded", self.data_len, "ys loaded", self.fake_len, "data faked")


    def __getitem__(self, index):
        data = self.ys[index % self.len]
        state = self.states[index % self.len]
        one_hot = self.one_hot[index % self.len]
        return data, state, one_hot

    def __add__(self, other):
        pass

    def __len__(self):
        return self.fake_len

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.len, size=batch_size)
        data = self.ys[indices]
        state = self.states[indices]
        one_hot = self.one_hot[indices]
        return data, state, one_hot

    def _load_data(self, args):
            dataset_path = args.dataset_path/ f'{args.env}'/f'{args.task}'/f'{args.dataset}'
            print("dataset_path:", dataset_path)
            dataset = np.load(dataset_path, allow_pickle='TRUE').item()
            data = {}
            data["states"] = dataset["observations"]
            data["actions"] = dataset["actions"]
            data["rewards"] = dataset["rewards"][:, None]
            data["done"] = dataset["dones"]
            data["returns"] = np.zeros((data["states"].shape[0], 1))
            last = 0
            for i in range(data["returns"].shape[0] - 1, -1, -1):
                last = data["rewards"][i, 0] + 0.99 * last * (1. - data["done"][i, 0])
                data["returns"][i, 0] = last
            return data

