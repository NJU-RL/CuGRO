import numpy as np

from . import register_env
from src.tp_envs.half_cheetah import HalfCheetahEnv


@register_env('cheetah-vel')
class HalfCheetahVelEnv(HalfCheetahEnv):

    def __init__(self, tasks=[{}], randomize_tasks=True):
        self.tasks = tasks
        self._task = self.tasks[0]
        self._goal_vel = self._task.get('velocity', 0.0)
        self._goal = self._goal_vel
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks, seed: int = 1337):
        np.random.seed(seed)
        #velocities = np.random.uniform(0.0, 3.0, size=(num_tasks,))
        velocities = np.linspace(0.075,3,40)
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()
