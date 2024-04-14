import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusion_SDE import dpm_solver_pytorch

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)

class SharedCriticNetwork(nn.Module):
    def __init__(
        self,
        sdim: int,
        adim: int,
        hdim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(sdim + adim, hdim),
            SiLU(),
            nn.Linear(hdim, hdim),
            SiLU(),
            nn.Linear(hdim, hdim),
            SiLU(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([state, action], dim=-1))

class MultiHeadCritic(nn.Module):
    def __init__(
        self,
        sdim: int,
        adim: int,
        hdim: int,
        num_heads: int,
        head_idx: int,
    ):
        super().__init__()
        self.shared_net = SharedCriticNetwork(sdim, adim, hdim)
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(nn.Sequential(
            nn.Linear(hdim, 1),)
                )
        self.head_idx = head_idx

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.shared_net(state, action)
        q_value = self.heads[self.head_idx](x)
        return q_value

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ScoreBase(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.device=args.device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.q = []
        self.q.append(MultiHeadCritic(sdim=input_dim-output_dim, adim=output_dim, hdim=256, num_heads=args.num_tasks, head_idx=args.task-1) if args.critic_type == "small" else MultiHeadCritic(sdim=input_dim-output_dim, adim=output_dim, hdim=512, num_heads=args.num_tasks, head_idx=args.task-1))
        self.args = args

    def forward_dmp_wrapper_fn(self, x, t):
        score = self(x, t)
        result = - score * self.marginal_prob_std(t)[1][:, None]
        return result

    # diffusion sample
    def dpm_wrapper_sample(self, dim, batch_size, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()

    def gan_sample(self, noise, cond):
        with torch.no_grad():
            samples = self.forward_generator(noise, cond)
            return samples

    def calculateQ(self, s, a, t=None):
        if s is None:
            if self.condition.shape[0] == a.shape[0]:
                s = self.condition
            elif self.condition.shape[0] == 1:
                s = torch.cat([self.condition]*a.shape[0])
            else:
                assert False
        return self.q[0](s,a)

    def calculateQ_new(self, s, a, task, t=None):
        if s is None:
            if self.condition.shape[0] == a.shape[0]:
                s = self.condition
            elif self.condition.shape[0] == 1:
                s = torch.cat([self.condition]*a.shape[0])
            else:
                assert False
        q_list = []
        for i in range(task-1):
            self.q[0].head_idx = i
            q_list.append(self.q[0](s[i],a[i]))
        self.q[0].head_idx = task-1
        return q_list

    def forward(self, x, t, condition=None):
        raise NotImplementedError

    def _select(self, returns, actions, alpha, num=1, replace=True):
        # returns: (n, 4) actions (n, NA)
        returns = returns[:, 0]
        returns = returns * alpha
        allowed_max = np.sort(returns)[-num] + 40
        returns[returns > allowed_max] = allowed_max
        unnormalised_p = np.exp(returns - np.max(returns))
        p = unnormalised_p / np.sum(unnormalised_p)
        index = np.random.choice(actions.shape[0], p=p, size=num, replace=replace)
        selected_returns = returns[index]
        selected_returns = np.exp(selected_returns - np.max(selected_returns))
        self.weighted = selected_returns / np.sum(selected_returns)
        return actions[index]

    def select_actions(self, states, sample_per_state=32, select_per_state=1, alpha=100, replace=False, weighted_mean=False, diffusion_steps=25, Normalizer=None):
        returns, actions = self.sample(states, sample_per_state, diffusion_steps, Normalizer=Normalizer)
        if isinstance(select_per_state, int):
            select_per_state = [select_per_state] * actions.shape[0]
        if (isinstance(alpha, int) or isinstance(alpha, float)):
            alpha = [alpha] * actions.shape[0]
        if (isinstance(replace, int) or isinstance(replace, float) or isinstance(replace, bool)):
            replace = [replace] * actions.shape[0]
        if (isinstance(weighted_mean, int) or isinstance(weighted_mean, float) or isinstance(weighted_mean, bool)):
            weighted_mean = [weighted_mean] * actions.shape[0]
        # select `select_per_sample` data from 32 data, ideally should be 1.
        # Selection should happen according to `alpha`
        # replace defines whether to put back data
        out_actions = []
        for i in range(actions.shape[0]):
            raw_actions = self._select(returns[i], actions[i], alpha=alpha[i], num=select_per_state[i], replace=replace[i])
            out_actions.append(np.average(raw_actions, weights=self.weighted if weighted_mean[i] else None, axis=0))
        return out_actions

    def sample(self, states, sample_per_state=16, diffusion_steps=25, Normalizer=None):
        self.eval()
        num_states = states.shape[0]
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            # t = time.time()
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=3)
            results = results.clip(-1, 1)
            if Normalizer is not None:
                states = torch.FloatTensor(Normalizer.unnormalize(np.array(states.to("cpu").numpy()), 'states')).to(self.device)
                results = Normalizer.unnormalize(np.array(results), 'actions')
                returns = self.calculateQ(states, torch.FloatTensor(results).to(self.device)).reshape(num_states, sample_per_state, 1).to("cpu").detach().numpy()
            else:
                returns = self.calculateQ(states, torch.FloatTensor(results).to(self.device)).reshape(num_states, sample_per_state, 1).to("cpu").detach().numpy()
            actions = results[:, :].reshape(num_states, sample_per_state, self.output_dim).copy()
            self.condition = None
        self.train() # train
        return returns, actions

    def action_sample(self, states, sample_per_state=16, diffusion_steps=25):
        self.eval()
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=3)
        self.train() # train
        return results.clip(-1, 1)


class GenerateBase(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.device=args.device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.args = args

    def forward_dmp_wrapper_fn(self, x, t, w = 3.0):
        score = self(x, t)
        result = - score * self.marginal_prob_std(t)[1][:, None]
        return result

    def dpm_wrapper_sample(self, dim, batch_size, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()

    def forward(self, x, t, condition=None):
        raise NotImplementedError

    def state_sample(self, states, sample_per_state=16, diffusion_steps=25):
        self.eval()
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=3)
        self.train() # train
        return results.clip(-1, 1)

class Residual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim=128, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.cond_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim),SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim),SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x, t, c):
        h1 = self.dense1(x) + self.time_mlp(t) + self.cond_mlp(c)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)


#残差模块
class Gene_Residual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim=128, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.cond_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim), SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim),SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x, t, c):
        h1 = self.dense1(x) + self.time_mlp(t) + self.cond_mlp(c)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)

class AttentionModule(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj_q = nn.Linear(in_dim, in_dim)
        self.proj_k = nn.Linear(in_dim, in_dim)
        self.proj_v = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        B, C = x.shape
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.view(B, 1, C)
        k = k.view(B, C, 1)
        w = torch.bmm(q, k) * (C ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.view(B, 1, C)
        h = torch.bmm(w, v)
        h = h.view(B, C)

        return h

#加了Atten的残差模块
class AttenResidual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, atten_dim, t_dim=512, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim),SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim),SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.attn = AttentionModule(atten_dim)

    def forward(self, x, t):
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        h3 = h2 + self.modify_x(x)
        return self.attn(h3)

class ScoreNet(ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim-output_dim, embed_dim), SiLU())
        self.sort_t = nn.Sequential(
                        nn.Linear(64, 128),
                        SiLU(),
                        nn.Linear(128, 128),
                    )
        self.down_block1 = Residual_Block(output_dim, 512)
        self.down_block2 = Residual_Block(512, 256)
        self.down_block3 = Residual_Block(256, 128)
        self.middle1 = Residual_Block(128, 128)
        self.up_block3 = Residual_Block(256, 256)
        self.up_block2 = Residual_Block(512, 512)
        self.last = nn.Linear(1024, output_dim)

    def forward(self, x, t, condition=None):
        embed_t = self.embed(t) # 对t编码
        if condition is not None:
            embed_c = self.pre_sort_condition(condition)
        else:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition]*x.shape[0])
            else:
                assert False
            embed_c = self.pre_sort_condition(condition)
        embed_t = self.sort_t(embed_t)
        embed_c = self.sort_t(embed_c)

        d1 = self.down_block1(x, embed_t, embed_c)
        d2 = self.down_block2(d1, embed_t, embed_c)
        d3 = self.down_block3(d2, embed_t, embed_c)
        u3 = self.middle1(d3, embed_t, embed_c)
        u2 = self.up_block3(torch.cat([d3, u3], dim=-1), embed_t, embed_c)
        u1 = self.up_block2(torch.cat([d2, u2], dim=-1), embed_t, embed_c)
        u0 = torch.cat([d1, u1], dim=-1)
        h = self.last(u0)
        self.h = h
        # Normalize output
        return h / self.marginal_prob_std(t)[1][:, None]

class GenerateNet(GenerateBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim-output_dim, embed_dim), SiLU())
        self.sort_t = nn.Sequential(
                        nn.Linear(64, 128),
                        SiLU(),
                        nn.Linear(128, 128),
                    )

        self.down_block1 = Gene_Residual_Block(output_dim, 512)
        self.down_block2 = Gene_Residual_Block(512, 256)
        self.down_block3 = Gene_Residual_Block(256, 128)
        self.middle1 = Gene_Residual_Block(128, 128)
        self.up_block3 = Gene_Residual_Block(256, 256)
        self.up_block2 = Gene_Residual_Block(512, 512)
        self.last = nn.Linear(1024, output_dim)

    def forward(self, x, t, condition=None):
        embed_t = self.embed(t)
        if condition is not None:
            embed_c = self.pre_sort_condition(condition)
        else:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition]*x.shape[0])
            else:
                assert False
            embed_c = self.pre_sort_condition(condition)

        embed_t = self.sort_t(embed_t)
        embed_c = self.sort_t(embed_c)

        d1 = self.down_block1(x, embed_t, embed_c)
        d2 = self.down_block2(d1, embed_t, embed_c)
        d3 = self.down_block3(d2, embed_t, embed_c)
        u3 = self.middle1(d3, embed_t, embed_c)
        u2 = self.up_block3(torch.cat([d3, u3], dim=-1), embed_t, embed_c)
        u1 = self.up_block2(torch.cat([d2, u2], dim=-1), embed_t, embed_c)
        u0 = torch.cat([d1, u1], dim=-1)

        h = self.last(u0)
        self.h = h
        # Normalize output
        return h / self.marginal_prob_std(t)[1][:, None]

class MlpScoreNet(ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        # self.act = lambda x: x * torch.sigmoid(x)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim-output_dim, embed_dim//2), SiLU())
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim//2),
            nn.Linear(embed_dim//2, embed_dim), SiLU(), nn.Linear(embed_dim, embed_dim//2), SiLU())
        self.block1 = nn.Sequential(
            nn.Linear(output_dim+embed_dim, 256),
            SiLU(),
            nn.Linear(256, 256),
            SiLU(),
            nn.Linear(256, 256),
            SiLU(),
        )
        self.decoder = Dense(256, output_dim)

    def forward(self, x, t, condition=None):
        embed_t = self.embed(t)
        if condition is not None:
            embed_c = self.pre_sort_condition(condition)
        else:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition]*x.shape[0])
            else:
                assert False
            embed_c = self.pre_sort_condition(condition)
        h = torch.cat((x, embed_t, embed_c),dim=-1)
        h = self.block1(h)
        h = self.decoder(h)
        # Normalize output
        h = h / self.marginal_prob_std(t)[1][:, None]
        return h

class MlpGenerateNet(GenerateBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim-output_dim, embed_dim//2), SiLU())
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim//2),
            nn.Linear(embed_dim//2, embed_dim), SiLU(), nn.Linear(embed_dim, embed_dim//2), SiLU())
        self.block1 = nn.Sequential(
            nn.Linear(output_dim+embed_dim, 256),
            SiLU(),
            nn.Linear(256, 256),
            SiLU(),
            nn.Linear(256, 256),
            SiLU(),
        )
        self.decoder = Dense(256, output_dim)

    def forward(self, x, t, condition=None):
        embed_t = self.embed(t)
        if condition is not None:
            embed_c = self.pre_sort_condition(condition)
        else:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition]*x.shape[0])
            else:
                assert False
            embed_c = self.pre_sort_condition(condition)
        # Encoding path
        h = torch.cat((x, embed_t, embed_c),dim=-1)
        h = self.block1(h)
        h = self.decoder(h)
        # Normalize output
        h = h / self.marginal_prob_std(t)[1][:, None]
        return h



