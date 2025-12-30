from Envs.RLDrone.BaseDroneEnv import BaseDroneEnv
from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch
from datetime import datetime
import os
import json
import math
import torchvision.utils as vutils
from Configs.RLDrone.Custom.Config import DataCollection_Config

class DataCollectionEnv(BaseDroneEnv):
    def __init__(self, VecTask_cfg, rl_device, sim_device, graphics_device_id, headless):
        super(DataCollectionEnv, self).__init__(VecTask_cfg, rl_device, sim_device, graphics_device_id, headless)
        self.data_collection = []
        self.config = DataCollection_Config

    def collect_data(self):
        # 1. 准备保存路径
        run_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(run_dir, exist_ok=True)

        b = self.boundaries
        low_limit = torch.tensor([b[0], b[2], self.config.height_low], device=self.device)
        high_limit = torch.tensor([b[1], b[3], self.config.height_high], device=self.device)

        fov_rad = math.radians(self.config.Camera.horizontal_fov)

        for step_idx in range(self.config.max_steps):
            # 初始化当前步的有效位姿缓存
            valid_found = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            final_pose = self.pose.clone()

            # 穿模检测重采样循环
            for attempt in range(self.config.max_attempts):
                if valid_found.all(): break
                
                if self.config.type == 'random' or step_idx == 0:
                    # 全局随机采样
                    test_pos = torch.rand((self.num_envs, 3), device=self.device) * (high_limit - low_limit) + low_limit
                
                elif self.config.type == 'nearby':
                    # 基于当前高度动态计算最大位移，确保相邻帧交集
                    # 覆盖半径 R = h * tan(FOV/2)，移动距离控制在 R 以内以保证重叠
                    current_h = self.pose[:, 2]
                    max_move = current_h * math.tan(fov_rad / 2) * 0.9 # 0.9倍确保边缘有交集
                    
                    offset = (torch.rand((self.num_envs, 3), device=self.device) * 2 - 1)
                    offset[:, 0:2] *= max_move.unsqueeze(-1)
                    offset[:, 2] *= self.config.height_change_scale # 高度波动范围
                    
                    # 应用边界约束，防止飞出场景边界
                    test_pos = torch.clamp(self.pose[:, :3] + offset, low_limit, high_limit)

                # 生成随机 Pitch (俯仰) 和 Yaw (偏航)
                # Pitch: 在垂直向下(pi/2)基础上增加随机偏移
                test_pitch = (torch.pi / 2) + (torch.rand(self.num_envs, device=self.device) - 0.5) * (math.radians(self.config.pitch_range_deg))
                test_yaw = torch.rand(self.num_envs, device=self.device) * 2 * torch.pi
                test_rot = torch.stack([torch.zeros_like(test_yaw), test_pitch, test_yaw], dim=-1)
                
                test_pose = torch.cat([test_pos, test_rot], dim=-1)
                
                # 预演位姿并更新视觉传感器以获取深度图
                self.update_pose(test_pose)
                self.update_visual()
                
                # --- 避障判定逻辑 ---
                # 1. 深度检测：最近距离过近说明撞墙或在物体表面
                min_depth, _ = torch.min(self.depth_processed.view(self.num_envs, -1), dim=-1)
                # 2. 空间开放度：如果最大深度也很小，说明在封闭物体内部
                max_depth, _ = torch.max(self.depth_processed.view(self.num_envs, -1), dim=-1)
                
                # 有效性判定：距离物体至少0.5m且视野中存在超过3m的开放空间
                is_valid = (min_depth > 0.5) & (max_depth > 3.0)
                
                # 更新尚未找到有效位置的环境位姿
                mask = is_valid & (~valid_found)
                final_pose[mask] = test_pose[mask]
                valid_found[mask] = True

            # 应用最终确定的安全位姿并渲染
            self.update_pose(final_pose)
            self.update_visual()
            
            # 遍历每个环境保存数据
            for env_idx in range(self.num_envs):
                # 获取数据
                rgb = self.rgb_processed[env_idx]   # [3, H, W]
                depth = self.depth_processed[env_idx] # [H, W]
                seg = self.seg_processed[env_idx]   # [H, W]
                
                # 保存图像 (按要求命名)
                vutils.save_image(rgb.float() / 255.0, f"{run_dir}/rgb_{env_idx}_{step_idx}.png")
                vutils.save_image(depth / self.config.Camera.depth_range, f"{run_dir}/depth_{env_idx}_{step_idx}.png")
                vutils.save_image(seg.float() / self._next_seg_id, f"{run_dir}/seg_{env_idx}_{step_idx}.png")

                # 记录索引信息
                self.data_collection.append({
                    "step": step_idx,
                    "env_idx": env_idx,
                    "pose": self.pose[env_idx].tolist(),
                    "rgb_path": f"rgb_{env_idx}_{step_idx}.png",
                    "depth_path": f"depth_{env_idx}_{step_idx}.png",
                    "seg_path": f"seg_{env_idx}_{step_idx}.png"
                })

        # 5. 保存 JSON 索引
        with open(f"{run_dir}/dataset_index.json", 'w') as f:
            json.dump(self.data_collection, f, indent=4)
        print(f"Data collection finished. Saved to {run_dir}")