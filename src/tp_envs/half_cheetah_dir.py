import numpy as np

from src.tp_envs.half_cheetah import HalfCheetahEnv
from . import register_env


@register_env('cheetah-dir')
class HalfCheetahDirEnv(HalfCheetahEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=False):
        directions = [-1, 1]
        self.tasks = [{'direction': direction} for direction in directions]
        self._task = task
        self._goal_dir = task.get('direction', 1)
        self._goal = self._goal_dir
        super(HalfCheetahDirEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_dir = self._task['direction']
        self._goal = self._goal_dir
        self.reset()
