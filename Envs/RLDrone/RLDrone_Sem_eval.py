from Envs.RLDrone.SemanticOccDrone import SemanticOccDrone
from Envs.utils import *
import torch
import numpy as np
from collections import deque
from Configs.RLDrone.Custom.Config import RL_Drone_Sem_eval_Config
from gym.spaces import Box, Dict, MultiDiscrete
from Envs.RLDrone.RLDrone_Sem import RLDrone_Sem

class RLDrone_Sem_eval(RLDrone_Sem):

    def __init__(self, 
            VecTask_cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless,
            **kwargs):
        
        # 使用eval_num_envs覆盖num_envs
        VecTask_cfg['env']['numEnvs'] = \
            RL_Drone_Sem_eval_Config.eval_num_envs

        super().__init__(
            VecTask_cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless,
            **kwargs)

        self.config = RL_Drone_Sem_eval_Config

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
        last_coverage_rate = self.height_coverage_buf[-1]
        coverage = (last_coverage_rate >= self.config.height_coverage_threshold)
        self.reset_buf |= coverage

        # NOTE: add ohter termination conditions here

    def calculate_reward(self):
        '''
        计算奖励函数
        '''
        # height coverage reward
        coverage_reward = self.height_coverage_buf[-1] - self.height_coverage_buf[-2]
        self.episode_rewards['height_coverage'] += coverage_reward
        coverage_reward = coverage_reward * self.config.height_coverage_scale * self.dt

        # termination reward
        termination_reward = self.reset_buf * ~self.time_out_buf
        termination_reward = termination_reward * self.config.termination_scale * self.dt

        # Penalty for current episode_length
        current_length = self.episode_length_buf.clone()
        extra_step = -torch.clip(
            current_length - self.config.short_path_threshold, 
            min=0, max=self.config.max_penalty_steps)
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