import os
from tqdm import tqdm, trange
import functools
import time
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, MlpScoreNet, GenerateNet, MlpGenerateNet
from utils import get_args
from dataset import Diffusion_buffer
import time
import json
from src.envs import  HalfCheetahVelEnv, WalkerRandParamsWrappedEnv, SwimmerDir
from collections import namedtuple
from utils import Config, get_optimizer, init_seeds, reduce_tensor, DataLoaderDDP
import yaml
import torch.distributed as dist
from continualworld.envs import get_cl_env
from continualworld.tasks import TASK_SEQS
from normalization import DatasetNormalizer

def Tasks(config):
    with open(config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    tasks = []
    for task_idx in (range(task_config.total_tasks)):
        with open(task_config.task_paths.format(task_idx*5), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
    return tasks

def one_hot_encode_task(task_index, num_tasks):
    encoding = np.zeros(num_tasks)
    encoding[task_index-1] = 1
    return encoding

def behavior(args):
    for dir in ["./models", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", args.env, args.data_mode+str(args.diffusion_steps), str(args.task))):
        os.makedirs(os.path.join("./models",args.env, args.data_mode+str(args.diffusion_steps), str(args.task)))

    if args.env == 'cheetah_vel':
        config="config/cheetah_vel/40tasks_offline.json"
        tasks = Tasks(config)
        env = HalfCheetahVelEnv(tasks)
    elif args.env == 'walker_params':
        config ="config/walker_params/50tasks_offline.json"
        tasks = Tasks(config)
        env = WalkerRandParamsWrappedEnv(tasks)
    elif args.env == 'swimmer_dir':
        config="config/swimmer_dir/50tasks_offline.json"
        tasks = Tasks(config)
        env = SwimmerDir(tasks = tasks)
    elif args.env == 'meta_world':
        tasks = TASK_SEQS['CL5']
        env = get_cl_env(tasks)
    else:
        raise RuntimeError(f'Invalid env name {args.env}')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    onehot_dim = args.num_tasks
    max_action = float(env.action_space.high[0])
    print("state_dim:", state_dim, "action_dim:", action_dim, "max_action:", max_action)
    env.seed(args.seed)

    yaml_path = args.config
    local_rank = args.local_rank
    use_amp = args.use_amp
    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print(opt)
    opt = Config(opt)
    print("local_rank:", local_rank)
    device = "cuda:%d" % local_rank
    print("device:", device)
    args.device = device
    args.critic_mode = False

    torch.manual_seed(args.seed+local_rank)
    np.random.seed(args.seed+local_rank)
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma, device=args.device) #保持参数不变
    args.marginal_prob_std_fn = marginal_prob_std_fn

    if args.actor_type == "large":
        score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
        generate_model= GenerateNet(input_dim=state_dim+onehot_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    elif args.actor_type == "small":
        score_model= MlpScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
        generate_model= MlpGenerateNet(input_dim=state_dim+onehot_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)

    save_path = os.path.join("./logs/", args.env, args.data_mode+str(args.diffusion_steps), str(args.task))
    if local_rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    score_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(score_model)
    score_model = torch.nn.parallel.DistributedDataParallel(
        score_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    generate_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generate_model)
    generate_model = torch.nn.parallel.DistributedDataParallel(
        generate_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    dataset = Diffusion_buffer(args)
    data_len = dataset.states.shape[0]
    Normalizer = DatasetNormalizer(dataset, 'LimitsNormalizer')
    dataset.states = Normalizer.normalize(dataset.data_["states"], "states")
    dataset.actions = Normalizer.normalize(dataset.data_["actions"], "actions")
    normalizers =Normalizer.normalizers
    normalizers_file = os.path.join(save_path, "normalizers.npy")
    with open(normalizers_file, "wb") as f:
        pickle.dump(normalizers, f)

    data_loader, sampler = DataLoaderDDP(dataset,
                                          batch_size= args.batch_size,
                                          shuffle=True)
    print("data_loader_len:", len(data_loader))

    lr = args.lr
    DDP_multiplier = dist.get_world_size()
    print("Using DDP, lr = %f * %d" % (lr, DDP_multiplier))
    lr *= DDP_multiplier
    score_optimizer = get_optimizer(score_model.parameters(), opt, lr=lr)
    generate_optimizer = get_optimizer(generate_model.parameters(), opt, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.task >= 2:
        if args.actor_load_setting is None:
            args.actor_loadpath = os.path.join("./models", str(args.env), args.data_mode+str(args.diffusion_steps), str(args.task-1), "ckpt{}.pth".format(args.
            actor_load_epoch))
            args.generate_loadpath = os.path.join("./models", str(args.env), args.data_mode+str(args.diffusion_steps), str(args.task-1), "generate_ckpt{}.pth".format(args.
            actor_load_epoch))
        else:
            args.actor_loadpath = os.path.join("./models", args.env + str(args.seed) + args.actor_load_setting, "ckpt{}.pth".format(args.actor_load_epoch))
            args.generate_loadpath = os.path.join("./models", args.env + str(args.seed) + args.actor_load_setting, "generate_ckpt{}.pth".format(args.actor_load_epoch))

        print("loading actor{}...".format(args.actor_loadpath))
        checkpoint = torch.load(args.actor_loadpath, map_location=args.device)
        score_model.load_state_dict(checkpoint['MODEL'])
        if args.data_mode == "gene":
            print("loading generate{}...".format(args.generate_loadpath))
            checkpoint = torch.load(args.generate_loadpath, map_location=args.device)
            generate_model.load_state_dict(checkpoint['MODEL'])


    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(os.path.join("./logs", args.env, args.data_mode+str(args.diffusion_steps), str(args.task), current_time))
        args.writer = writer

    if args.data_mode == "gene":
        if args.task >= 2:
            print("generate data")
            states_list, actions_list = [np.zeros((data_len, state_dim)) for i in range(args.task-1)], [np.zeros((data_len, action_dim)) for i in range(args.task-1)]
            generate_model.eval()
            score_model.eval()
            for i in range(1, args.task):
                states = np.zeros((data_len, state_dim))
                actions = np.zeros((data_len, action_dim))
                returns = np.zeros((data_len, 1))
                one_hot_list = np.zeros((data_len, dataset.one_hot.shape[1]))
                index = 0
                task_hot = one_hot_encode_task(i, args.num_tasks)
                for x, condition, _ in tqdm(data_loader, disable=False):
                    batch = condition.shape[0]
                    one_hot =np.repeat([task_hot], batch, axis = 0).astype(np.float32)
                    sample_method = generate_model.module.state_sample
                    generate_states = sample_method(one_hot, sample_per_state= 1, diffusion_steps=args.diffusion_steps)
                    sample_method = score_model.module.action_sample
                    generate_actions = sample_method(generate_states, sample_per_state= 1, diffusion_steps=args.diffusion_steps)
                    generate_actions  = torch.tensor(generate_actions).to(args.device)
                    generate_states  = torch.tensor(generate_states).to(args.device)
                    world_size = dist.get_world_size()
                    gathered_actions = [torch.zeros_like(generate_actions) for _ in range(world_size)]
                    gathered_states = [torch.zeros_like(generate_states) for _ in range(world_size)]
                    dist.all_gather(gathered_actions, generate_actions)
                    dist.all_gather(gathered_states, generate_states)
                    gathered_actions = torch.cat(gathered_actions, dim=0).cpu().numpy()
                    gathered_states = torch.cat(gathered_states, dim=0).cpu().numpy()
                    next_index = index + batch * world_size
                    if next_index >= data_len:
                        next_index = data_len
                    one_hot = np.repeat([task_hot], next_index - index, axis = 0).astype(np.float32)
                    states[index:next_index, :] = gathered_states[:next_index-index]
                    actions[index:next_index, :] = gathered_actions[:next_index-index]
                    returns[index:next_index, :] =  np.zeros((next_index-index, 1))
                    one_hot_list[index:next_index, :] =np.array(one_hot)
                    states_list[i-1][index:next_index, :] =  gathered_states[:next_index-index]
                    actions_list[i-1][index:next_index, :] = gathered_actions[:next_index-index]
                    index = next_index
                dataset.states = np.concatenate((dataset.states, states.astype(np.float32)), axis=0)
                dataset.actions = np.concatenate((dataset.actions, actions.astype(np.float32)), axis=0)
                dataset.returns = np.concatenate((dataset.returns, returns.astype(np.float32)), axis=0)
                dataset.ys = np.concatenate([dataset.returns, dataset.actions], axis=-1)
                dataset.one_hot = np.concatenate((dataset.one_hot, one_hot_list.astype(np.float32)))

        dataset.len = dataset.states.shape[0]
        dataset.fake_len = dataset.states.shape[0]
        print(dataset.len, "data loaded", dataset.fake_len, "data faked")

    data_loader, sampler = DataLoaderDDP(dataset,
                                          batch_size = args.batch_size,
                                          shuffle=True)
    print("data_loader_len:", len(data_loader))

    if args.task >= 1:
        print("training diffusion")
        behavior_loss = []
        tqdm_epoch = trange(opt.load_epoch + 1, args.n_behavior_epochs)
        for epoch in tqdm(tqdm_epoch, disable=(local_rank != 0)):
            avg_loss = 0.
            num_items = 0
            sampler.set_epoch(epoch)
            dist.barrier()
            score_model.train()
            pbar = data_loader
            for x, condition, _ in pbar:
                x = x[:, 1:]
                x = x.to(args.device)
                condition = condition.to(args.device)
                loss = loss_fn(score_model, x, args.marginal_prob_std_fn, condition)
                score_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(score_optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=score_model.parameters(), max_norm=1.0)
                scaler.step(score_optimizer)
                scaler.update()
                dist.barrier()
                loss = reduce_tensor(loss)
                avg_loss += loss * x.shape[0]
                num_items += x.shape[0]
            avg_loss = avg_loss.item() / num_items
            tqdm_epoch.set_description('Behavior Loss: {:5f}'.format(avg_loss))
            if local_rank == 0:
                if epoch % 100 == 99 and args.save_model:
                    checkpoint = {
                        'MODEL': score_model.state_dict(),
                        'opt': score_optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join("./models", str(args.env), args.data_mode+str(args.diffusion_steps), str(args.task), "ckpt{}.pth".format(epoch+1)))
                if args.writer:
                    args.writer.add_scalar("actor/loss", avg_loss, global_step=epoch)
                behavior_loss.append(avg_loss)
        if local_rank == 0:
            save_fig = os.path.join(save_path, "behavior_loss")
            plt.plot(np.arange(len(behavior_loss)),behavior_loss)
            plt.xlabel('behavior_loss')
            plt.ylabel('loss')
            plt.savefig(save_fig, dpi=300)
            plt.cla()
            print("behavior model finished")

    if args.task < args.num_tasks and args.data_mode == "gene":
        print("training generate")
        generate_loss = []
        tqdm_epoch = trange(opt.load_epoch + 1, args.n_behavior_epochs)
        for epoch in tqdm(tqdm_epoch):
            avg_loss = 0.
            num_items = 0
            sampler.set_epoch(epoch)
            dist.barrier()
            generate_model.train()
            pbar = data_loader
            for _, state, condition in pbar:
                state = state.to(args.device)
                condition = condition.to(args.device)
                loss = loss_fn(generate_model, state, args.marginal_prob_std_fn, condition)
                generate_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(generate_optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=generate_model.parameters(), max_norm=1.0)
                scaler.step(generate_optimizer)
                scaler.update()
                dist.barrier()
                loss = reduce_tensor(loss)
                avg_loss += loss * state.shape[0]
                num_items +=state.shape[0]
            avg_loss = avg_loss.item() / num_items
            tqdm_epoch.set_description('Generate Loss: {:5f}'.format(avg_loss))

            if local_rank == 0:
                if epoch % 100 == 99 and args.save_model:
                    checkpoint = {
                        'MODEL': generate_model.state_dict(),
                        'opt': generate_optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join("./models", str(args.env), args.data_mode+str(args.diffusion_steps), str(args.task), "generate_ckpt{}.pth".format(epoch+1)))
                if args.writer:
                    args.writer.add_scalar("generate/loss", avg_loss , global_step=epoch)
                generate_loss.append(avg_loss)
        if local_rank == 0:
            save_fig = os.path.join(save_path, "generate_loss")
            plt.plot(np.arange(len(generate_loss)),generate_loss)
            plt.xlabel('generate_loss')
            plt.ylabel('loss')
            plt.savefig(save_fig, dpi=300)
            plt.cla()
            print("generate model finishd")
    if local_rank == 0:
        writer.close()

if __name__ == "__main__":
    args = get_args()
    dist.init_process_group(backend='nccl')
    print("Available CUDA devices:", torch.cuda.device_count())
    print("current local_rank:", args.local_rank)
    init_seeds(no=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    t = time.time()
    for i in range(1, args.num_tasks+1):
        args.task = i
        behavior(args)
    print("total time:", (time.time()-t)/3600)
