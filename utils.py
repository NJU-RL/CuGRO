import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cheetah_vel")
    parser.add_argument("--dataset_path", default=Path("dataset"))
    parser.add_argument("--dataset", default=Path("dataset.npy"))
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--expid", default="default", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--save_model", default=1, type=int)
    parser.add_argument("--task", default=0, type=int)
    parser.add_argument("--num_tasks", default=5, type=int)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument("--gpu", default="0", help="Specify the GPU to use")
    parser.add_argument('--data_mode', type=str, default="gene")
    parser.add_argument('--critic_mode', type=bool, default=False)
    parser.add_argument('--sigma', type=float, default=40.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--alpha', type=float, default=20.0)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--n_behavior_epochs', type=int, default=600)
    parser.add_argument('--actor_load_epoch', type=int, default= 600)
    parser.add_argument('--normalise_return', type=int, default=1)
    parser.add_argument('--critic_type', type=str, default=None)
    parser.add_argument('--actor_type', type=str, default="large")
    parser.add_argument('--critic_load_epoch', type=int, default=99)
    parser.add_argument('--actor_load_setting', type=str, default=None)
    parser.add_argument('--critic_load_setting', type=str, default=None)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--seed_per_evaluation', type=int, default=10)
    parser.add_argument('--evaluate_while_training_critic', type=int, default=1)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--l', type=float, default=1)
    parser.add_argument("--config", type=str, default = "config/cifar_conditional.yaml")
    parser.add_argument('--local_rank', default= 0, type=int,
                        help='node rank for distributed training') #-1
    parser.add_argument("--use_amp", action='store_true', default=False)
    print("**************************")
    args = parser.parse_known_args()[0]
    if not ("cheetah" in args.env or "walker" in args.env or "swimmer" in args.env):
        args.select_per_state = 1
    else:
        args.select_per_state = 4 # stablize performance
    print(args)
    return args

def plot_tools(folder_name, env_name, task, return_list, all_return_list, data_mode, setting_name=None, seed=0, p=1):
    if isinstance(seed, list):
        ys = []
        stds = []
        for s in seed:
            ts, y, std = plot_tools(folder_name, setting_name, task, s, None)
            ys.append(y)
            stds.append(std)
        ys = np.stack(ys)
        stds = np.std(ys, axis=0)
        ys = np.mean(ys, axis=0)
        if p:
            plt.plot(ts, ys)
            plt.fill_between(ts, ys-stds, ys+stds, alpha=0.4)
        return ts, ys, stds
    else:
        tfevent_file = os.path.join(folder_name, env_name, data_mode, str(task))
        plt_path = os.path.join(tfevent_file, "returns")
        for i in range(task):
            plt_list = np.array(return_list[i])
            ts = plt_list[:,2]
            ys = plt_list[:,0]
            stds = plt_list[:,1]
            plt.plot(ts,ys, label='task_{}'.format(i+1))
            plt.fill_between(ts, ys-stds, ys+stds, alpha=0.4)
        plt.xlabel('{}.episode'.format(env_name))
        plt.ylabel('Return')
        plt.legend()
        plt.savefig(plt_path, dpi=300)
        plt.cla()
        return ts, ys, stds

def plot_successs(folder_name, env_name, task, return_list, all_return_list, data_mode, setting_name=None, seed=0, p=1):
    if isinstance(seed, list):
        ys = []
        stds = []
        for s in seed:
            ts, y, std = plot_tools(folder_name, setting_name, task, s, None)
            ys.append(y)
            stds.append(std)
        ys = np.stack(ys)
        stds = np.std(ys, axis=0)
        ys = np.mean(ys, axis=0)
        if p:
            plt.plot(ts, ys)
            plt.fill_between(ts, ys-stds, ys+stds, alpha=0.4)
        return ts, ys, stds
    else:
        tfevent_file = os.path.join(folder_name, env_name, data_mode, str(task))
        plt_path = os.path.join(tfevent_file, "success_rate")
        for i in range(task):
            plt_list = np.array(return_list[i])
            ts = plt_list[:,2]
            ys = plt_list[:,0]
            stds = plt_list[:,1]
            plt.plot(ts,ys, label='task_{}'.format(i+1))
            plt.fill_between(ts, ys-stds, ys+stds, alpha=0.4)
        plt.xlabel('{}.episode'.format(env_name))
        plt.ylabel('Success_rate')
        plt.legend()
        plt.savefig(plt_path, dpi=300)
        plt.cla()
        return ts, ys, stds

# ===== Configs =====

class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])

def get_optimizer(parameters, opt, lr):
    if not hasattr(opt, 'optim'):
        return torch.optim.Adam(parameters, lr=lr)
    elif opt.optim == 'AdamW':
        return torch.optim.AdamW(parameters, **opt.optim_args, lr=lr)
    else:
        raise NotImplementedError()

# ===== Multi-GPU training =====

def init_seeds(RANDOM_SEED=1337, no=0):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def gather_tensor(tensor):
    tensor_list = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list

def DataLoaderDDP(dataset, batch_size, shuffle=True):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=1,
    )
    return dataloader, sampler

def print0(*args, **kwargs):
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        print(*args, **kwargs)


def schedules(betas, T, device, type='DDPM'):
    beta1, beta2 = betas
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), torch.linspace(beta1, beta2, T)])
    elif type == 'DDIM':
        beta_t = torch.linspace(beta1, beta2, T + 1)
    else:
        raise NotImplementedError()
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

    dic = {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "ma_over_sqrtmab": ma_over_sqrtmab,
    }
    return {key: dic[key].to(device) for key in dic}
