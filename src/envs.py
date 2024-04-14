import numpy as np
from typing import Optional, Tuple, List
from src.tp_envs.half_cheetah_vel import HalfCheetahVelEnv as HalfCheetahVelEnv_
from src.tp_envs.half_cheetah_dir import HalfCheetahDirEnv as HalfCheetahDirEnv_
from src.tp_envs.ant_dir import AntDirEnv as AntDirEnv_
from src.tp_envs.ant_goal import AntGoalEnv as AntGoalEnv_
from src.tp_envs.humanoid_dir import HumanoidDirEnv as HumanoidDirEnv_
from src.tp_envs.walker_rand_params_wrapper import WalkerRandParamsWrappedEnv as WalkerRandParamsWrappedEnv_
from gym.spaces import Box
from gym.wrappers import TimeLimit
from copy import deepcopy

import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env




class ML45Env(object):
    def __init__(self, include_goal: bool = False):
        self.n_tasks = 50
        self.tasks = list(HARD_MODE_ARGS_KWARGS['train'].keys()) + list(HARD_MODE_ARGS_KWARGS['test'].keys())

        self._max_episode_steps = 150

        self.include_goal = include_goal
        self._task_idx = None
        self._env = None
        self._envs = []

        _cls_dict = {**HARD_MODE_CLS_DICT['train'], **HARD_MODE_CLS_DICT['test']}
        _args_kwargs = {**HARD_MODE_ARGS_KWARGS['train'], **HARD_MODE_ARGS_KWARGS['test']}
        for idx in range(self.n_tasks):
            task = self.tasks[idx]
            args_kwargs = _args_kwargs[task]
            if idx == 28 or idx == 29:
                args_kwargs['kwargs']['obs_type'] = 'plain'
                args_kwargs['kwargs']['random_init'] = False
            else:
                args_kwargs['kwargs']['obs_type'] = 'with_goal'
            args_kwargs['task'] = task
            env = _cls_dict[task](*args_kwargs['args'], **args_kwargs['kwargs'])
            self._envs.append(TimeLimit(env, max_episode_steps=self._max_episode_steps))

        self.set_task_idx(0)

    @property
    def observation_space(self):
        space = self._env.observation_space
        if self.include_goal:
            space = Box(low=space.low[0], high=space.high[0], shape=(space.shape[0] + len(self.tasks),))
        return space

    def reset(self):
        obs = self._env.reset()
        if self.include_goal:
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[self._task_idx] = 1.0
            obs = np.concatenate([obs, one_hot])
        return obs

    def step(self, action):
        o, r, d, i = self._env.step(action)
        if self.include_goal:
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[self._task_idx] = 1.0
            o = np.concatenate([o, one_hot])
        return o, r, d, i

    def set_task_idx(self, idx):
        self._task_idx = idx
        self._env = self._envs[idx]

    def __getattribute__(self, name):
        '''
        If we try to access attributes that only exist in the env, return the
        env implementation.
        '''
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            e_ = e
            try:
                return object.__getattribute__(self._env, name)
            except AttributeError as env_exception:
                pass
            except Exception as env_exception:
                e_ = env_exception
        raise e_

class HalfCheetahDirEnv(HalfCheetahDirEnv_):
    def __init__(self, tasks: List[dict], include_goal: bool = False):
        self.include_goal = include_goal
        self.current_step = 0
        self._max_episode_steps = 200
        super(HalfCheetahDirEnv, self).__init__()
        if tasks is None:
            tasks = [{'direction': 1}, {'direction': -1}]
        self.tasks = tasks
        self.set_task_idx(0)


    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.current_step += 1  # Increment the step counter
        if self.current_step >= self._max_episode_steps:
            done = True  # Set done=True after reaching the maximum number of steps
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0  # Reset the step counter to 0 at the beginning of each episode
        return super().reset()

    def set_task(self, task):
        self._task = task
        self._goal_dir = self._task['direction']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])


class HopperFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, tasks: List[dict] = None, include_goal: bool = False, one_hot_goal: bool = False, n_tasks: int = None):
        asset_path = os.path.join(
            os.path.dirname(__file__), 'tp_envs/assets/hopper.xml')
        utils.EzPickle.__init__(self)
        self.include_goal = include_goal
        self.one_hot_goal = one_hot_goal
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            # tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.current_step = 0
        self._max_episode_steps = 200
        self._task = self.tasks[0]
        self._goal_vel = self._task.get('velocity', 0.0)
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        # alive_bonus = 1.0
        # forward_vel = (posafter - posbefore) / self.dt
        # reward = -1.0 * abs(forward_vel - self._goal_vel)
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        alive_bonus = 1.0
        forward_vel = (posafter - posbefore) / self.dt
        forward_reward = - 4.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = - 0.01 * np.square(a).sum()
        reward = forward_reward + ctrl_cost + alive_bonus
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        self.current_step += 1
        if self.current_step >= self._max_episode_steps:
            done = True  # Set done=True after reaching the maximum number of steps
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            # self.sim.data.qpos.flat[1:],
            self.sim.data.qpos.flat,
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset(self):
        self.current_step = 0  # Reset the step counter to 0 at the beginning of each episode
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['velocity']
        self.reset()

    def set_task_idx(self, idx):
        self.task_idx = idx
        self.set_task(self.tasks[idx])



class SwimmerDir(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="swimmer.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        include_goal: bool = False,
        tasks: List[dict]= None,
        one_hot_goal: bool = False,
        n_tasks: int = None,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
            )

        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.current_step = 0
        self._max_episode_steps = 200
        self._task = self.tasks[0]
        self._goal = self._task.get('goal', 0.0)
        # mujoco_env.MujocoEnv.__init__(
        #     self, xml_file, 4, observation_space=observation_space, **kwargs
        # )
        mujoco_env.MujocoEnv.__init__(
            self, xml_file, 4)
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.sim.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        direct = (np.cos(self._goal), np.sin(self._goal))
        xy_position_after = self.sim.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # forward_reward = self._forward_reward_weight * x_velocity
        forward_reward = np.dot((xy_velocity/self.dt), direct)


        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        self.current_step += 1
        if self.current_step >= self._max_episode_steps:
            done = True  # Set done=True after reaching the maximum number of steps
        else:
            done = False
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        # if self.render_mode == "human":
        #     self.render()

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

    def reset(self):
        self.current_step = 0  # Reset the step counter to 0 at the beginning of each episode
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def set_task(self, task):
        self._task = task
        self._goal= self._task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])

class HalfCheetahVelEnv(HalfCheetahVelEnv_):
    def __init__(self, tasks: List[dict] = None, include_goal: bool = False, one_hot_goal: bool = False, n_tasks: int = None):
        self.include_goal = include_goal
        self.one_hot_goal = one_hot_goal
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.n_tasks = len(tasks)
        self.current_step = 0
        self._max_episode_steps = 200
        super().__init__(tasks)
        self.set_task_idx(0)

    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            if self.one_hot_goal:
                goal = np.zeros((self.n_tasks,))
                goal[self.tasks.index(self._task)] = 1
            else:
                goal = np.array([self._goal_vel])
            obs = np.concatenate([obs, goal])
        else:
            obs = super()._get_obs()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.current_step += 1  # Increment the step counter
        if self.current_step >= self._max_episode_steps:
            done = True  # Set done=True after reaching the maximum number of steps
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0  # Reset the step counter to 0 at the beginning of each episode
        return super().reset()

    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['velocity']
        self.reset()

    def set_task_idx(self, idx):
        self.task_idx = idx
        self.set_task(self.tasks[idx])

class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: List[dict], n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        self.current_step = 0
        self._max_episode_steps = 200
        super(AntDirEnv, self).__init__(forward_backward=n_tasks == 2)
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.set_task_idx(0)


    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(50, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.current_step += 1  # Increment the step counter
        if self.current_step >= self._max_episode_steps:
            done = True  # Set done=True after reaching the maximum number of steps
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0  # Reset the step counter to 0 at the beginning of each episode
        return super().reset()

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])


######################################################
######################################################
# <BEGIN DEPRECATED> #################################
######################################################
######################################################
class AntGoalEnv(AntGoalEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False,
                 reward_offset: float = 0.0, can_die: bool = False):
        self.include_goal = include_goal
        self.reward_offset = reward_offset
        self.can_die = can_die
        super().__init__()
        if tasks is None:
            tasks = self.sample_tasks(130) #Only backward-forward tasks
        self.tasks = tasks
        self._task = tasks[task_idx]
        self.task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self.info_dim = 2

    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            obs = np.concatenate([obs, self._goal])
        else:
            obs = super()._get_obs()
        return obs

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()

class HumanoidDirEnv(HumanoidDirEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False):
        self.include_goal = include_goal
        super(HumanoidDirEnv, self).__init__()
        if tasks is None:
            tasks = self.sample_tasks(130) #Only backward-forward tasks
        self.tasks = tasks
        self._task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self.info_dim = 1

    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            obs = np.concatenate([obs, np.array([np.cos(self._goal), np.sin(self._goal)])])
        else:
            obs = super()._get_obs()
        return obs

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if done == True:
            rew = rew - 5.0
            done = False
        return (obs, rew, done, info)

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()
######################################################
######################################################
# </END DEPRECATED> ##################################
######################################################
######################################################

class WalkerRandParamsWrappedEnv(WalkerRandParamsWrappedEnv_):
    def __init__(self, tasks: List[dict] = None, n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        self.n_tasks = len(tasks) if tasks is not None else n_tasks
        self.current_step = 0
        self._max_episode_steps = 200
        super(WalkerRandParamsWrappedEnv, self).__init__(tasks, n_tasks)
        self.set_task_idx(0)


    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self._goal
            except:
                pass
            one_hot = np.zeros(self.n_tasks, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.current_step += 1  # Increment the step counter
        if self.current_step >= self._max_episode_steps:
            done = True  # Set done=True after reaching the maximum number of steps
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0  # Reset the step counter to 0 at the beginning of each episode
        return super().reset()


    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()


