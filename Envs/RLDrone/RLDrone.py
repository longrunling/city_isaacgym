from Envs.RLDrone.OccMapDrone import OccMapDrone
from Envs.RLDrone.SemanticOccDrone import SemanticOccDrone
from Envs.utils import *
import torch
import numpy as np
from collections import deque
from Configs.RLDrone.Custom.Config import RLDrone_Config
from gym.spaces import Box, Dict, MultiDiscrete

class RLDrone(OccMapDrone):

    def __init__(self, 
            VecTask_cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless,
            **kwargs):

        super().__init__(
            VecTask_cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless,
            **kwargs)

        self.config = RLDrone_Config

        self._create_obs_act_space()
        self._create_buffers_RL()

    def _create_obs_act_space(self):
        '''
        获取观测空间与动作空间
        设置了VecTask的属性
        self.num_actions, self.act_space, self.obs_space
        '''
        # 动作空间
        action_space_size = (self.pose_up_idx - self.pose_low_idx + 1).cpu().numpy()
        self.action_size = action_space_size.shape[0]
        self.num_actions = self.action_size

        '''
        使用self.action_space调用
        '''
        # For rl_games
        # actions = ()
        # for size in action_space_size:
        #     actions += (Discrete(size), )        
        # self.act_space = Tuple(actions)

        # For SB3
        self.act_space = MultiDiscrete(nvec=torch.Size(action_space_size))

        # 观测空间
        pose_up = self.get_pose_from_pose_idx(
            self.pose_up_idx).cpu().numpy().reshape(1, -1)
        pose_up = pose_up.repeat(self.buffer_size, axis=0).astype(np.float32)
        pose_low = self.get_pose_from_pose_idx(
            self.pose_low_idx).cpu().numpy().reshape(1, -1)
        pose_low = pose_low.repeat(self.buffer_size, axis=0).astype(np.float32)

        state_space = Box(low=pose_low, high=pose_up,
            shape=(self.buffer_size, self.action_size), dtype=np.int64)
        occ_map_space = Box(low=-torch.inf, high=torch.inf,
            shape=tuple(self.occ_map_size.tolist()), dtype=np.float32)

        '''
        使用self.observation_space调用
        '''
        self.obs_space = Dict({
            'state': state_space,
            'occ_map': occ_map_space
        })

        # for network input shape
        self.input_shape = {
            'occ_map': tuple(self.occ_map_size.tolist()),
            'state': (self.buffer_size, self.action_size)
        }        

    def _create_buffers_RL(self):
        
        # 创建轨迹长度缓存
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)

        # 创建是否重置缓存
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        
        # 创建覆盖率buff
        self.coverage_buf = deque(maxlen=max(self.buffer_size, 2))
        self.coverage_buf.extend(max(self.buffer_size, 2) * [torch.zeros(self.num_envs, device=self.device)])

        # 创建总奖励缓存
        self.reward_keys = ['coverage', 'short_path', 'sum']
        self.episode_rewards = {
            name: torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_keys
        }
        
        # for SB3
        self.rewbuffer = {
            name: deque(maxlen=100) for name in self.reward_keys
        }
        self.lenbuffer = deque(maxlen=100)
        self.time_out_buf = (self.episode_length_buf >= self.buffer_size)
        self.extras = {}
        
    def reset_idx(self, env_idx):
        '''
        重置env_idxs指定的环境
        '''
        num_envs = len(env_idx)
        if num_envs == 0:
            return

        # 必须调用父类的reset_envs
        super().reset_idx(env_idx)

        # 重置是否重置缓存
        self.reset_buf[env_idx] = 0

        # 重置覆盖率缓存
        for buf_idx in range(self.buffer_size):
            self.coverage_buf[buf_idx][env_idx] = 0

        # 重置总奖励缓存
        for name in self.reward_keys:
            self.episode_rewards[name][env_idx] = 0.

        # 记录是否因为时间步数用尽而重置
        self.extras["time_outs"] = self.time_out_buf

        # 重置轨迹长度缓存
        self.episode_length_buf[env_idx] = 0        

    def update_coverage_rate(self):
        '''
        计算覆盖率
        '''
        covered_voxels = torch.sum(self.scanned_gt_occ, dim=(1, 2, 3))
        coverage_rate = covered_voxels / self.num_valid_voxels_gt
        self.coverage_buf.append(coverage_rate)

    def get_observations(self):
        '''
        更新所有状态
        '''
        # 获取新的视觉观测
        self.update_visual()

        # 更新占据地图
        self.update_occ_map()
        # self.update_occ_map_gennbv()
        # self.update_occ_grid()

        # 更新覆盖率
        self.update_coverage_rate()

        state = torch.stack(tuple(self.pose_buf),dim=1)
        rgb = self.rgb_processed.clone()
        occ_map = self.occ_map_cls.clone()

        return {
            'state': state,
            # 'rgb': rgb,
            'occ_map': occ_map
        }
    
    def reset(self):

        # 重置所有环境
        env_idx = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(env_idx)

        # 使用初始位姿作为初始动作
        init_pose = self.init_pose.repeat(self.num_envs, 1)
        self.update_pose(init_pose)

        # 获取初始观测
        obs = self.get_observations()

        # 更新轨迹长度
        self.episode_length_buf += 1   

        # 计算奖励
        _ = self.calculate_reward()

        # 更新额外的结束信息
        dones = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.update_extra_episode_info(dones)                      

        # 重置所有环境
        env_idx = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(env_idx)        

        # self.render()

        return obs

    def step(self, actions):

        self.actions = actions.clone()

        # 获取上一次被重置的环境的索引，被重置的环境episode_length_buf为0
        env_idxs = [idx for idx in range(self.num_envs) 
                    if self.episode_length_buf[idx] == 0]
        # 重置这些环境的动作为初始动作
        if len(env_idxs) != 0:
            self.actions[env_idxs] = \
            self.init_pose_idx.repeat(self.num_envs, 1)[env_idxs]

        # 获取动作对应的位姿
        pose = self.get_pose_from_pose_idx(self.actions)

        # 更新位置
        self.update_pose(pose)

        # 更新所有状态
        obs = self.get_observations()

        # 更新轨迹长度
        self.episode_length_buf += 1        

        # 检查是否需要重置环境
        self.check_termination()

        # 完成标志
        dones = self.reset_buf.clone()

        # 计算奖励
        reward = self.calculate_reward()

        # 更新额外的结束信息
        self.update_extra_episode_info(dones)

        # 计算需要重置的环境索引并重置
        env_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_idx)

        # self.render()

        return obs, reward, dones, self.extras
    
    def check_termination(self):
        """ 
        Check if environments need to be reset
        """
        # max_step
        time_out = (self.episode_length_buf >= self.buffer_size)
        self.reset_buf |= time_out
        self.time_out_buf = time_out

        # coverage rate
        last_coverage_rate = self.coverage_buf[-1]
        coverage = (last_coverage_rate >= self.config.coverage_threshold)
        self.reset_buf |= coverage

        # NOTE: add ohter termination conditions here

    def calculate_reward(self):
        '''
        计算奖励函数
        '''
        # surface coverage reward
        coverage_reward = self.coverage_buf[-1] - self.coverage_buf[-2]
        self.episode_rewards['coverage'] += coverage_reward
        coverage_reward = coverage_reward * self.config.coverage_scale * self.dt

        # termination reward
        termination_reward = self.reset_buf * ~self.time_out_buf
        termination_reward = termination_reward * self.config.termination_scale * self.dt

        # Penalty for current episode_length
        current_length = self.episode_length_buf.clone()
        extra_step = -torch.clip(current_length - 30, min=0, max=2)
        self.episode_rewards['short_path'] += extra_step
        extra_step = extra_step * self.config.short_path_scale * self.dt
        # reward is computed cumulatively

        # 记录总奖励
        reward = coverage_reward + extra_step
        self.episode_rewards['sum'] += reward

        return reward
    
    def update_extra_episode_info(self, dones):
        '''
        用于记录结束的环境的平均奖励与平均长度
        '''
        new_ids = (dones > 0).nonzero(as_tuple=False)
        for name in self.reward_keys:
            self.rewbuffer[name].extend(self.episode_rewards[name][new_ids][:, 0].cpu().detach().numpy().tolist())
        self.lenbuffer.extend(self.episode_length_buf[new_ids][:, 0].cpu().detach().numpy().tolist())

        # 记录
        self.extras["episode"] = {}
        for name in self.reward_keys:
            self.extras["episode"][f'episode_reward_{name}'] = \
                np.mean(self.rewbuffer[name]) if len(self.rewbuffer[name]) > 0 else 0.
        self.extras["episode"]["episode_length"] = \
            np.mean(self.lenbuffer) if len(self.lenbuffer) > 0 else 0.

    def get_privileged_observations(self):
        pass