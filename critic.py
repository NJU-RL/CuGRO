from copy import deepcopy
import os
from tqdm import tqdm, trange
import functools
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, MlpScoreNet, GenerateNet, MlpGenerateNet
from utils import get_args, plot_tools, plot_successs
from dataset import Diffusion_buffer
import json
from src.envs import HalfCheetahVelEnv, WalkerRandParamsWrappedEnv, SwimmerDir
from collections import namedtuple
from continualworld.envs import get_cl_env
from continualworld.tasks import TASK_SEQS
from normalization import DatasetNormalizer

def one_hot_encode_task(task_index, num_tasks):
    encoding = np.zeros(num_tasks)
    encoding[task_index-1] = 1
    return encoding

def train_critic(args, score_model, cur_score_model, generate_model, data_loader, states_list, actions_list, optimizer, Normalizer, start_epoch=0):
    n_epochs = args.K *100
    tqdm_epoch = trange(start_epoch, n_epochs)

    old_q_net = deepcopy(score_model.q[0])
    loss_list = []
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        if args.evaluate_while_training_critic and ((epoch % 10 == 9) or epoch==0):
            # evaluation
            cur_score_model.load_state_dict(score_model.state_dict())
            cur_score_model.q[0].load_state_dict(score_model.q[0].state_dict())
            for i in range(args.task):
                cur_score_model.q[0].head_idx = i
                normalizers_file = os.path.join("./logs/", args.env, args.data_mode+str(args.diffusion_steps), str(i+1), "normalizers.npy")
                with open(normalizers_file, "rb") as f:
                    gene_normalizers = pickle.load(f)
                Normalizer.normalizers = gene_normalizers
                envs = args.eval_func(cur_score_model.select_actions, task= i+1, Normalizer = Normalizer)
                env_returns = [envs[i].dbag_return for i in range(args.seed_per_evaluation)]
                mean = np.mean(env_returns)
                std = np.std(env_returns)
                if args.env == "meta_world":
                    env_success = [envs[i].pop_successes() for i in range(args.seed_per_evaluation)]
                    success_mean = np.mean(env_success)
                    success_std = np.std(env_success)
                    success_list[i].append(np.array([success_mean, success_std, epoch + (args.task-1)*n_epochs]))
                    all_success_list[i].append(np.array(env_success))
                    print("success/rew:", success_mean, "success/std", success_std)
                if args.writer:
                    args.writer.add_scalar("eval/rew", mean, global_step=epoch)
                    args.writer.add_scalar("eval/std", std, global_step=epoch)
                print("eval/rew:", mean, "eval/std", std)
                return_list[i].append(np.array([mean, std, epoch + (args.task-1)*n_epochs]))
                all_return_list[i].append(np.array(env_returns))

        for x, condition, _ in data_loader:
            x = x.to(args.device)
            condition = condition.to(args.device)
            loss = torch.mean((score_model.calculateQ(condition, x[:,1:]) - x[:,:1])**2)
            if args.task >= 2:
                old_states, old_actions, old_q  = [],[],[]
                for i in range(1, args.task):
                    old_q_net.head_idx = i -1
                    random_indices = np.random.choice(states_list[i-1].shape[0], condition.shape[0], replace=False)
                    generate_states = states_list[i-1][random_indices, :]
                    generate_actions = actions_list[i-1][random_indices, :]
                    generate_states = torch.tensor(generate_states, dtype=torch.float32, device=args.device)
                    generate_actions = torch.tensor(generate_actions, dtype=torch.float32, device=args.device)
                    generate_q = old_q_net(generate_states, generate_actions)
                    old_states.append(generate_states)
                    old_actions.append(generate_actions)
                    old_q.append(generate_q)
                new_q = score_model.calculateQ_new(old_states, old_actions, args.task)
                for i in range(len(new_q)):
                    dist_loss = nn.functional.mse_loss(new_q[i], old_q[i])
                    loss = loss + args.l * dist_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            score_model.condition = None
            avg_loss += loss * x.shape[0]
            num_items += x.shape[0]
        avg_loss = avg_loss.item() / num_items
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss))

        if (epoch % 50 == 48) or epoch==0:
            torch.save(score_model.q[0].state_dict(), os.path.join("./models", str(args.env), args.data_mode+str(args.diffusion_steps), str(args.task), "critic_ckpt{}.pth".format(epoch+1)))

        if args.writer:
            args.writer.add_scalar("critic/loss", avg_loss, global_step=epoch)
        loss_list.append(avg_loss)
    return loss_list


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

def pallaral_eval_policy(policy_fn, env_name, task, seed, eval_episodes=20, track_obs=False, select_per_state=1, diffusion_steps=15, Normalizer = None):
    del track_obs
    eval_envs = []
    for i in range(eval_episodes):
        if env_name == 'cheetah_vel':
            config="config/cheetah_vel/40tasks_offline.json"
            tasks = Tasks(config)
            env = HalfCheetahVelEnv(tasks)
            env.set_task_idx(task-1)
        elif env_name == 'walker_params':
            config ="config/walker_params/50tasks_offline.json"
            tasks = Tasks(config)
            env = WalkerRandParamsWrappedEnv(tasks)
            env.set_task_idx(task-1)
        elif args.env == 'swimmer_dir':
            config="config/swimmer_dir/50tasks_offline.json"
            tasks = Tasks(config)
            env = SwimmerDir(tasks = tasks)
            env.set_task_idx(task-1)
        elif args.env == 'meta_world':
            tasks = TASK_SEQS['CL5']
            env = get_cl_env(tasks)
            env.set_task_idx(task-1)
        else:
            raise RuntimeError(f'Invalid env name {env_name}')

        eval_envs.append(env)
        env.seed(seed + 1001 + i)
        env.dbag_state = env.reset()
        env.dbag_return = 0.0
        env.alpha = 100 # 100 could be considered as deterministic sampling since it's now extremely sensitive to normalized Q(s, a)
        env.select_per_state = select_per_state
    ori_eval_envs = [env for env in eval_envs]
    t = time.time()
    while len(eval_envs) > 0:
        new_eval_envs = []
        states = np.stack([env.dbag_state for env in eval_envs])
        states = Normalizer.normalize(states, 'states')
        actions = policy_fn(states, sample_per_state=32, select_per_state=[env.select_per_state for env in eval_envs], alpha=[env.alpha for env in eval_envs], replace=False, weighted_mean=False, diffusion_steps=diffusion_steps,Normalizer=Normalizer)
        # actions = Normalizer.unnormalize(np.array(actions), 'actions')
        for i, env in enumerate(eval_envs):
            state, reward, done, info = env.step(actions[i])
            env.dbag_return += reward
            env.dbag_state = state
            if not done:
                new_eval_envs.append(env)
        eval_envs = new_eval_envs
    print("time:", time.time() - t)
    return ori_eval_envs

def critic(args):
    for dir in ["./models", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models", args.env, args.data_mode+str(args.diffusion_steps), str(args.task))):
        os.makedirs(os.path.join("./models", args.env,  args.data_mode+str(args.diffusion_steps), str(args.task)))
    if not os.path.exists(os.path.join("./logs/" + str(args.l), args.env, args.data_mode+str(args.diffusion_steps), str(args.task))):
        os.makedirs(os.path.join("./logs/" + str(args.l), args.env, args.data_mode+str(args.diffusion_steps), str(args.task)))

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

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, task=args.task, seed=args.seed, eval_episodes=args.seed_per_evaluation, track_obs=False, select_per_state=args.select_per_state, diffusion_steps=args.diffusion_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    onehot_dim = args.num_tasks
    max_action = float(env.action_space.high[0])
    print("state_dim:", state_dim, "action_dim:", action_dim, "max_action:", max_action)

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    if args.actor_type == "large":
        score_model = ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
        cur_score_model = ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
        generate_model = GenerateNet(input_dim=state_dim+onehot_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)

    elif args.actor_type == "small":
        score_model = MlpScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
        cur_score_model = MlpScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
        generate_model = MlpGenerateNet(input_dim=state_dim+onehot_dim, output_dim=state_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)

    score_model.q[0].to(args.device) #critic network
    cur_score_model.q[0].to(args.device) #old critic network
    if args.actor_load_setting is None:
        args.actor_loadpath = os.path.join("./models", str(args.env),  args.data_mode+str(args.diffusion_steps), str(args.task), "ckpt{}.pth".format(args.actor_load_epoch))
        args.generate_loadpath = os.path.join("./models", str(args.env),  args.data_mode+str(args.diffusion_steps), str(args.task-1), "generate_ckpt{}.pth".format(args.actor_load_epoch))
    else:
        args.actor_loadpath = os.path.join("./models", args.env + str(args.seed) + args.actor_load_setting, "ckpt{}.pth".format(args.actor_load_epoch))
        args.generate_loadpath = os.path.join("./models", args.env + str(args.seed) + args.actor_load_setting, "generate_ckpt{}.pth".format(args.actor_load_epoch))

    print("loading actor{}...".format(args.actor_loadpath))
    checkpoint = torch.load(args.actor_loadpath, map_location=args.device)
    new_state_dict = {k[7:]: v for k, v in checkpoint['MODEL'].items()}
    score_model.load_state_dict(new_state_dict)
    if args.task >= 2 and args.data_mode == "gene" :
        print("loading generate{}...".format(args.generate_loadpath))
        checkpoint = torch.load(args.generate_loadpath, map_location=args.device)
        new_state_dict = {k[7:]: v for k, v in checkpoint['MODEL'].items()}
        generate_model.load_state_dict(new_state_dict)
    #lodaing multihead critic network
    if args.task >= 2:
        if args.critic_load_setting is None:
            args.critic_loadpath = os.path.join("./models", str(args.env), args.data_mode+str(args.diffusion_steps), str(args.task-1), "critic_ckpt{}.pth".format(args.critic_load_epoch))
        else:
            args.critic_loadpath = os.path.join("./models", args.env + str(args.seed) + args.actor_load_setting, "critic_ckpt{}.pth".format(args.critic_load_epoch))
        print("loading critic{}...".format(args.critic_loadpath))
        ckpt = torch.load(args.critic_loadpath, map_location=args.device)
        score_model.q[0].load_state_dict(ckpt)

    args.critic_mode = True
    dataset = Diffusion_buffer(args)
    Normalizer = DatasetNormalizer(dataset, 'LimitsNormalizer')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    data_len = dataset.states.shape[0]
    actions_list = []
    states_list = []
    t = time.time()
    if args.task >= 2 and args.data_mode == "gene":
        print("************generate states and actions**************")
        states_list, actions_list = [np.zeros((data_len, state_dim)) for i in range(args.task-1)], [np.zeros((data_len, action_dim)) for i in range(args.task-1)]
        for i in range(1, args.task):
            normalizers_file = os.path.join("./logs/", args.env, args.data_mode+str(args.diffusion_steps), str(i), "normalizers.npy")
            with open(normalizers_file, "rb") as f:
                gene_normalizers = pickle.load(f)
            Normalizer.normalizers = gene_normalizers
            task_hot = one_hot_encode_task(i, args.num_tasks)
            index = 0
            for _, condition, _ in tqdm(data_loader, disable=False):
                one_hot =np.repeat([task_hot], condition.shape[0], axis = 0).astype(np.float32)
                sample_method = generate_model.state_sample
                generate_states = sample_method(one_hot, sample_per_state= 1, diffusion_steps=args.diffusion_steps)
                sample_method = score_model.action_sample
                generate_actions = sample_method(generate_states, sample_per_state= 1, diffusion_steps=args.diffusion_steps)
                states_list[i-1][index:index+condition.shape[0], :] =  generate_states
                actions_list[i-1][index:index+condition.shape[0], :] = generate_actions
                index = index + condition.shape[0]
            print("norm_state:", states_list[i-1].max(), states_list[i-1].min())
            print("norm_action:", actions_list[i-1].max(), actions_list[i-1].min())
            states_list[i-1] = Normalizer.unnormalize(np.array(states_list[i-1]), 'states')
            actions_list[i-1] = Normalizer.unnormalize(np.array(actions_list[i-1]), 'actions')
            print("unnorm_state:", actions_list[i-1].max(), states_list[i-1].min())
            print("unnorm_action:", actions_list[i-1].max(), actions_list[i-1].min())
        print("generate states and actions time:{}".format(time.time()-t))


    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(os.path.join("./logs", args.env, args.data_mode+str(args.diffusion_steps), str(args.task), current_time))
    args.writer = writer

    print("training critic")
    optimizer = Adam(score_model.q[0].parameters(), lr=1e-3)
    critic_loss = train_critic(args, score_model, cur_score_model, generate_model, data_loader, states_list, actions_list, optimizer, Normalizer, start_epoch=0)

    save_path = os.path.join("./logs/" + str(args.l), args.env, args.data_mode+str(args.diffusion_steps), str(args.task))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_fig = os.path.join(save_path, "critic_loss")
    plt.plot(np.arange(len(critic_loss)),critic_loss)
    plt.xlabel('critic_loss')
    plt.ylabel('loss')
    plt.savefig(save_fig, dpi=300)
    plt.cla()
    print("critic model finished")

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global return_list
    global all_return_list
    global success_rate_list

    return_list = [[] for _ in range(args.num_tasks)]
    all_return_list = [[] for _ in range(args.num_tasks)]
    if args.env == "meta_world":
        success_list = [[] for _ in range(args.num_tasks)]
        all_success_list = [[] for _ in range(args.num_tasks)]

    # 序列化并保存列表到文件
    def save_list_to_file(file_path, my_list):
        with open(file_path, 'wb') as f:
            pickle.dump(my_list, f)

    # 从文件中加载并反序列化列表
    def load_list_from_file(file_path):
        with open(file_path, 'rb') as f:
            my_list = pickle.load(f)
        return my_list

    for dir in ["./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    file_path = os.path.join("./logs/" + str(args.l), args.env, args.data_mode+str(args.diffusion_steps))
    for i in range(1, args.num_tasks+1):
        args.task = i
        if args.task >=2:
            return_file = os.path.join(file_path, str(args.task-1), "return_list.npy")
            return_list = load_list_from_file(return_file)
            all_return_file = os.path.join(file_path, str(args.task-1), "all_return_list.npy")
            all_return_list = load_list_from_file(all_return_file)
            if args.env == "meta_world":
                success_file = os.path.join(file_path, str(args.task-1), "success_list.npy")
                success_list = load_list_from_file(success_file)
                all_success_file = os.path.join(file_path, str(args.task-1), "all_success_list.npy")
                all_success_list = load_list_from_file(all_success_file)

        critic(args)

        return_file = os.path.join(file_path, str(args.task), "return_list.npy")
        save_list_to_file(return_file, return_list)
        all_return_file = os.path.join(file_path, str(args.task), "all_return_list.npy")
        save_list_to_file(all_return_file, all_return_list)
        if args.env == "meta_world":
            success_file = os.path.join(file_path, str(args.task), "success_list.npy")
            save_list_to_file(success_file, success_list)
            all_success_file = os.path.join(file_path, str(args.task), "all_success_list.npy")
            save_list_to_file(all_success_file, all_success_list)

        plot_tools(folder_name= "./logs/" + str(args.l), env_name = args.env, task = args.task, return_list = return_list, all_return_list = all_return_list, data_mode = args.data_mode+str(args.diffusion_steps))
        if args.env == "meta_world":
            plot_successs(folder_name= "./logs/" + str(args.l), env_name = args.env, task = args.task, return_list = success_list, all_return_list = all_success_list, data_mode = args.data_mode+str(args.diffusion_steps))





