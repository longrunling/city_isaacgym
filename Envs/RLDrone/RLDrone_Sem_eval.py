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

        # 返回覆盖率
        height_coverage = self.height_coverage_buf[-1]

        return obs, reward, dones, self.extras, height_coverage