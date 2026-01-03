import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
from isaacgym import gymapi, gymtorch
from torchvision.transforms.functional import rgb_to_grayscale
from Configs.RLDrone.Custom.Config import BaseDroneEnv_Config
import torch
from isaacgym.torch_utils import *
from Envs.utils import class_to_dict
from collections import deque
import open3d as o3d
import re
import gym
from isaacgymenvs.tasks.base.vec_task import VecTask
import cv2
import json
import math
import torch_scatter
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from Envs.utils import *

URDF_SEG_ID = 255

class BaseDroneEnv(VecTask):

    def __init__(self,
            VecTask_cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless, 
            **kwargs):
        '''
        基础无人机环境类
        VecTask_cfg为必须设置的值
        rl_device, sim_device, graphics_device_id会在多gpu训练时使用外部函数设置
        '''
        self.cfg = VecTask_cfg
        # 获取环境生成器
        self.simple_citygen: simple_citygen = \
            kwargs.get('simple_citygen', None)        

        # 读取自定义配置
        self.config = BaseDroneEnv_Config
        
        super().__init__(
            self.cfg, 
            rl_device, 
            sim_device, 
            graphics_device_id, 
            headless)
        
    def create_sim(self):

        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt * 4

        # 绑定环境生成器
        self.simple_citygen.attach_env(self)

        self._create_envs()

    def _create_envs(self):

        self._create_lights()

        # 环境相关变量
        self.envs = []
        self.envs_actors_num = []
        self.envs_actors = []
        self.cameras = []
        self.drone_root_state_idx = []

        # 动作相关变量
        self.pose_low = torch.tensor(self.config.pose_low, dtype=torch.float32, device=self.device, requires_grad=False)
        self.pose_low_idx = torch.tensor(self.config.pose_low_idx, dtype=torch.int64, device=self.device, requires_grad=False)
        self.pose_up_idx = torch.tensor(self.config.pose_up_idx, dtype=torch.int64, device=self.device, requires_grad=False)
        self.pose_unit = torch.tensor(self.config.pose_unit, dtype=torch.float32, device=self.device, requires_grad=False)
        self.init_pose_idx = torch.tensor(self.config.init_pose_idx, dtype=torch.int64, device=self.device, requires_grad=False)

        # 根据动作空间计算环境坐标上限(可根据具体环境修改)
        self.env_min_cord = self.pose_low[0:3].repeat(self.num_envs, 1)

        pose_up = self.get_pose_from_pose_idx(self.pose_up_idx)
        env_max_cord = pose_up[0:3]
        env_max_cord = env_max_cord.repeat(self.num_envs, 1)
        self.env_max_cord = env_max_cord                

        # # 创建地面平面
        # plane_params = gymapi.PlaneParams()
        # plane_params.normal = gymapi.Vec3(0, 0, 1)
        # self.gym.add_ground(self.sim, plane_params)

        # 获取环境原点
        self.simple_citygen._get_env_origins()

        # 创建num_envs个环境
        for i in tqdm(range(self.num_envs), desc='Creating Environments'):

            min_vec = gymapi.Vec3(0, 0, 0)
            max_vec = gymapi.Vec3(0, 0, 0)

            env = self.gym.create_env(self.sim, 
                min_vec, max_vec,
                int(np.sqrt(self.num_envs)))
            
            # 初始化环境actor列表
            self.envs.append(env)
            self.envs_actors_num.append(0)
            self.envs_actors.append({})

            # 创建无人机和相机传感器
            self._create_drones(i)

            # 记录无人机状态索引编号
            # 编号为当前所有环境的actor数量之和减去1
            total_actor_num = sum(self.envs_actors_num)
            self.drone_root_state_idx.append(total_actor_num - 1)            

            # simple_citygen方式
            self.simple_citygen.additional_create(i)

        self.drone_root_state_idx = torch.tensor(
            self.drone_root_state_idx, dtype=torch.int64, device=self.device, requires_grad=False)
        
        self.class_num = self.simple_citygen._next_class_seg_id

    def _create_lights(self):

        directions = [
            gymapi.Vec3(0, 1, 0),
            gymapi.Vec3(1, 0, 0),
        ]
        for i, dir in enumerate(directions):
            self.gym.set_light_parameters(self.sim, i, 
                gymapi.Vec3(0.8, 0.8, 0.8), 
                gymapi.Vec3(0.8, 0.8, 0.8), 
                dir)
            
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
        spacing = self.config.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _get_world_pose_from_env_pose(self, env_pose, env_idx = None):
        '''
        获取相对全局环境的动作
        '''
        if env_pose.dim() == 1:
            assert env_idx is not None, "env_idx must be provided for single env_pose"
            world_pose = env_pose + self.env_origins[env_idx]
        else:
            world_pose = env_pose + self.env_origins
        return world_pose
    
    def _create_actor(self, env_idx, actor, name, segmentation_id = 0, pose = None):

        if pose is None:
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 0)
            pose.r = gymapi.Quat(0, 0, 0, 1)

        # 计算环境偏移
        p = torch.tensor([pose.p.x, pose.p.y, pose.p.z], device=self.device)
        world_p = self._get_world_pose_from_env_pose(p, env_idx)
        pose.p = gymapi.Vec3(world_p[0].item(), world_p[1].item(), world_p[2].item())

        # 创建Actor
        ahandle = self.gym.create_actor(
            self.envs[env_idx], 
            actor, pose, name, env_idx, 0)
        
        # 设置分割ID
        body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_idx], ahandle, 0)
        self.gym.set_rigid_body_segmentation_id(
            self.envs[env_idx], body_handle, 0, segmentation_id)        

        # 更新环境的actor数量和字典
        self.envs_actors_num[env_idx] += 1
        ahandle_dict = self.envs_actors[env_idx]
        ahandle_dict[name] = ahandle
        return ahandle    

    def _create_drones(self, env_idx):

        # 配置相机传感器（用于获取深度图）
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.config.Camera.width
        camera_props.height = self.config.Camera.height
        camera_props.near_plane = 0.0010000000474974513
        camera_props.far_plane = 2000000.0      
        camera_props.horizontal_fov = self.config.Camera.horizontal_fov
        camera_props.supersampling_horizontal = 1
        camera_props.supersampling_vertical = 1
        camera_props.enable_tensors = True

        # 创建代理
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.density = 0
        asset_options.collapse_fixed_joints = False
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0
        asset_options.linear_damping = 0
        asset_options.armature = 0
        asset_options.thickness = 0
        drone = self.gym.create_sphere(self.sim, 0, asset_options)

        # 需要修改为根据初始位置创建
        init_state = torch.tensor(self.config.init_state, device=self.device)
        position = (init_state[:3] + self.env_origins[env_idx]).tolist()
        quat = init_state[3:].tolist()
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*position)
        pose.r = gymapi.Quat(*quat)

        # drone = self._create_actor(env_idx, drone, 'drone', pose=pose)
        drone = self.gym.create_actor(
            self.envs[env_idx], 
            drone, 
            pose, 
            'drone', env_idx, 1, 0
        )
        # 更新环境的actor数量和字典
        self.envs_actors_num[env_idx] += 1
        ahandle_dict = self.envs_actors[env_idx]
        ahandle_dict['drone'] = drone
        
        # 创建相机传感器
        body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_idx], drone, 0)        
        camera = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)
        camera_offset = gymapi.Vec3(0, 0, 0)
        camera_rotation = gymapi.Quat(0, 0, 0, 1)
        self.gym.attach_camera_to_body(
            camera,
            self.envs[env_idx], 
            body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_TRANSFORM
        )
        self.cameras.append(camera)

        # 设置分割ID
        segmentation_id = self.envs_actors_num[env_idx]
        self.gym.set_rigid_body_segmentation_id(
            self.envs[env_idx], body_handle, 0, segmentation_id)

    def _create_gennbv_urdf(self, env_idx, data_folder, file_name):

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        data_folder_urdf = os.path.join(data_folder, 'urdf')        
        asset = self.gym.load_asset(self.sim, data_folder_urdf, file_name, asset_options)

        # 需要修改为根据初始位置创建
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.env_origins[env_idx][0],
                             self.env_origins[env_idx][1],
                             self.env_origins[env_idx][2])
        pose.r = gymapi.Quat(0, 0, 0, 1)

        # urdf = self._create_actor(env_idx, asset, file_name, pose=pose)
        # 创建Actor
        urdf = self.gym.create_actor(
            self.envs[env_idx], 
            asset, pose, file_name, env_idx, 0)
        # self.gym.set_actor_scale(self.envs[env_idx], urdf, scale_factor)
        # 更新环境的actor数量和字典
        self.envs_actors_num[env_idx] += 1
        ahandle_dict = self.envs_actors[env_idx]
        ahandle_dict[file_name] = urdf        

        # 设置分割ID
        body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_idx], urdf, 0)
        self.gym.set_rigid_body_segmentation_id(
            self.envs[env_idx], body_handle, 0, URDF_SEG_ID)
        
    def _create_buffers_Base(self):
        '''
        创建环境和动作相关的缓冲区
        '''
        # 最长轨迹长度
        self.max_episode_length = self.config.buffer_size

        # 获取初始状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(root_state)
        # 用于重置环境
        self.init_root_state = self.root_state.clone()

        # 存储历史位姿的缓冲区
        self.init_pose = self.get_pose_from_pose_idx(self.init_pose_idx)
        pose_buf = self.init_pose.repeat(self.num_envs, 1)
        self.buffer_size = self.config.buffer_size
        self.pose_buf = deque(maxlen=self.buffer_size)
        self.pose_buf.extend(self.buffer_size * [pose_buf])

        # 创建视觉访问区
        self._create_visual_buffers()

    def _create_visual_buffers(self):

        # 视觉访问区
        self.rgb_cam_tensors = []
        self.depth_cam_tensors = []
        self.seg_cam_tensors = []        

        # RGB buffer
        for i in range(self.num_envs):
            im = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_COLOR)
            torch_cam_tensor = gymtorch.wrap_tensor(im)
            self.rgb_cam_tensors.append(torch_cam_tensor)

        # Depth buffer
        for i in range(self.num_envs):
            im = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(im)
            self.depth_cam_tensors.append(torch_cam_tensor)

        # Segmentation buffer
        for i in range(self.num_envs):
            im = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_SEGMENTATION)
            torch_cam_tensor = gymtorch.wrap_tensor(im)
            self.seg_cam_tensors.append(torch_cam_tensor)

    def reset_idx(self, env_idx):
        '''
        重置env_idxs指定的环境
        '''
        num_envs = len(env_idx)
        if num_envs == 0:
            return

        # 重置drone位置
        reset_idx = self.drone_root_state_idx[env_idx]
        self.root_state[reset_idx, :] = self.init_root_state[reset_idx, :]

        # 随机初始化线速度和角速度
        # self.root_state[reset_idx, 7:] = torch_rand_float(
        #     -0.5, 0.5, (len(reset_idx), 6), device=self.device)  
        self.root_state[reset_idx, 7:] = 0
        # [7:10]: lin vel, [10:13]: ang vel

        reset_idx = reset_idx.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state), 
            gymtorch.unwrap_tensor(reset_idx),
            num_envs
        )

        # 重置动作缓存
        for buf_idx in range(self.buffer_size):
            self.pose_buf[buf_idx][env_idx] = self.init_pose.repeat(num_envs, 1)  

    def update_visual(self):
        '''
        更新rgb图像,灰度图,深度图像,分割图像
        rgb:self.rgb_processed
        gray:self.rgb_grayscale
        depth:self.depth_processed
        seg:self.seg_processed
        '''
        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)        
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        rgb_images = torch.stack(self.rgb_cam_tensors)[..., :3].permute(0, 3, 1, 2) 
        # [num_env, 3, H, W]
        self.rgb_processed = rgb_images.to(torch.int32)
        self.rgb_grayscale = rgb_to_grayscale(rgb_images).to(torch.float32)   
        # [num_env, 1, H, W], weighted

        depth_images = torch.stack(self.depth_cam_tensors)
        depth_images = torch.nan_to_num(depth_images, neginf=0)
        depth_images = abs(depth_images)
        depth_images = torch.clamp(depth_images, max=self.config.Camera.depth_range)

        # 将值为depth_range设为depth_range
        depth_images[depth_images >= self.config.Camera.depth_range] = 0.0

        self.depth_processed = depth_images 
        # [num_env, H, W]

        seg_images = torch.stack(self.seg_cam_tensors)
        seg_images = torch.nan_to_num(seg_images, neginf=0)
        self.seg_processed = seg_images

        self.gym.end_access_image_tensors(self.sim)

    def get_pose_from_pose_idx(self, pose_idx):
        '''
        获取相对环境的动作,并非世界坐标
        '''
        pose = pose_idx * self.pose_unit + self.pose_low
        return pose
    
    def update_pose(self, pose):
        '''
        设置无人机全局坐标
        '''
        position = pose[..., 0:3]
        heading = pose[..., 3:]

        # set position, consider global environment offset
        world_position = self._get_world_pose_from_env_pose(position)
        self.root_state[self.drone_root_state_idx, 0:3] = world_position
        # self.root_state[self.drone_root_state_idx, :].cpu().numpy()
        # set roll pitch yaw
        quat = quat_from_euler_xyz(heading[..., 0], heading[..., 1], heading[..., 2])
        self.root_state[self.drone_root_state_idx, 3:7] = quat
        self.root_state[self.drone_root_state_idx, 7:] = 0
        # set
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_state))
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # 将动作存入缓存
        self.pose_buf.append(pose)
        self.pose = pose

    def render_forever(self):

        while not self.gym.query_viewer_has_closed(self.viewer):

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

    def visualize_depth_image(self, env_idx):

        depth_image = self.depth_processed[env_idx].cpu().numpy()
        plt.figure()
        plt.imshow(depth_image, cmap='gray')
        plt.title('Depth Image')
        plt.axis('off')
        plt.colorbar()
        plt.savefig('depth_image.png')
        plt.close()  

    def visualize_rgb_image(self, env_idx):

        rgb_image = self.rgb_processed[env_idx].permute(1, 2, 0).cpu().numpy()
        plt.figure()
        plt.imshow(rgb_image)
        plt.title('RGB Image')
        plt.axis('off')
        plt.savefig('rgb_image.png') 
        plt.close()

    def visualize_rgb_image_cv2(self, env_idx):

        rgb_image = self.rgb_processed[env_idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('rgb_image_cv2.png', rgb_image)
  
    def visualize_seg_image(self, env_idx):

        seg_image = self.seg_processed[env_idx].cpu().numpy()
        plt.figure()
        plt.imshow(seg_image, cmap='gray')
        plt.title('Segmentation Image')
        plt.axis('off')
        plt.savefig('seg_image.png')
        plt.close()

    def get_house_3k_obj_name(self, urdf_name):
        '''
        从urdf文件名获取对应的3k obj文件名
        '''
        # 匹配所有数字
        numbers = re.findall(r'\d+', urdf_name)

        # 提取前两个数字（batch1 中的 1 和 setA_4 中的 4）
        num1, num2 = map(int, numbers[:2])

        # obj文件模板
        obj_name = f"BAT{num1}_SETA_HOUSE{num2}.obj"

        return obj_name
    
    # 接管VecTask的方法
    def post_physics_step(self):
        pass

    def pre_physics_step(self, actions):
        pass

    def allocate_buffers(self):
        self._create_buffers_Base()

class simple_citygen():

    def __init__(self):
        '''
        start_idx用于设置从第几个placement开始放置
        可用于区分train和eval环境
        '''
        self.placements = self._parse_placements()

        # segmentation id 映射：相同 obj/urdf 使用相同 seg id
        self._class_seg_id = {}

        # 记录class编号
        self._next_class_seg_id = 0

        # 资产缓存，避免重复 load
        self._asset_cache = {}
        self._obj_paths = []

        # 记录已创建的环境编号
        self._created_env_idx = 0

        # 记录已获取的点云
        self.num_samples = -1

    def copy_env(self, env_gen):
        '''
        复制另一个实例的信息
        用于多进程前
        '''
        self._class_seg_id = env_gen._class_seg_id.copy()
        self._next_class_seg_id = env_gen._next_class_seg_id
        self._created_env_idx = env_gen._created_env_idx

    def clear(self):
        '''
        清除资产目录
        保留已创建编号
        '''
        self.env = None
        self.config = None
        self.device = None
        self._asset_cache = {}
        self._obj_paths = []
        self._asset_cache_pytorch3d = None

    def attach_env(self, BaseDroneEnv: BaseDroneEnv):
        '''
        在创建前用于绑定环境
        '''
        self.env = BaseDroneEnv
        self.config = self.env.config
        self.device = self.env.device
        # 记录环境的基准索引,为上一次创建的环境数量
        self._env_base_idx = self._created_env_idx

    def _parse_placements(self):
        """尝试读取 placements.json 并返回条目列表，如果不存在返回空列表"""
        try_paths = []
        # 项目内默认 simple_citygen/data 路径
        repo_scene_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'simple_citygen', 'data', 'scene', 'placements.json'))
        try_paths.append(repo_scene_path)

        for p in try_paths:
            if os.path.exists(p):
                try:
                    with open(p, 'r') as f:
                        data = json.load(f)
                        print(f"[BaseDroneEnv] loaded placements from {p}, {len(data)} entries")
                        return data
                except Exception as e:
                    print(f"[BaseDroneEnv] failed to parse placements file {p}: {e}")
        return []

    def _resolve_urdf_path(self, src):
        """从 placements 中的 src 尝试解析出 (urdf_dir, urdf_name)。
           支持绝对路径或根据 data_folder 搜索同名文件。
        """
        if not src:
            return None, None

        # 如果是绝对路径且存在
        if os.path.isabs(src) and os.path.exists(src):
            return os.path.dirname(src), os.path.basename(src)

        basename = os.path.basename(src)
        repo_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'simple_citygen', 'data'))
        for root, dirs, files in os.walk(repo_data_root):
            if basename in files:
                return root, basename
            
        return None, None
        
    def additional_create(self, env_idx):
        '''
        地面分割id:0, 其余从1开始
        '''
        # 在每个 env 上创建一个大地面（box primitive），并上色为纯色，放在 z=0 平面
        try:
            ground_key = 'ground_box'
            if ground_key not in self._asset_cache:
                # 计算地面尺寸：基于 placements bbox 或 fallback
                if hasattr(self, 'placements') and self.placements:
                    xs = [
                        p['position'][0] for p in self.placements if 'position' in p and len(p['position']) >= 2]
                    ys = [
                        p['position'][1] for p in self.placements if 'position' in p and len(p['position']) >= 2]
                    if xs and ys:
                        width = max(xs) - min(xs)
                        length = max(ys) - min(ys)
                        # 使用配置中的 padding (已在 _get_env_origins 中引入) 来扩展边界
                        # padding = getattr(self.config, 'placement_padding', 10.0)
                        # width_padded = width + 2.0 * padding
                        # length_padded = length + 2.0 * padding
                        # ground_x = max(10.0, width_padded)
                        # ground_y = max(10.0, length_padded)
                        ground_x = max(10.0, width) + self.config.ground_padding
                        ground_y = max(10.0, length) + self.config.ground_padding
                    else:
                        ground_x = ground_y = 200.0
                else:
                    ground_x = ground_y = 200.0

                ground_z = 0.01
                asset_options = gymapi.AssetOptions()
                asset_options.flip_visual_attachments = True
                asset_options.fix_base_link = True
                asset_options.disable_gravity = True
                # create_box(signature may vary by isaacgym version)
                ground_asset = self.env.gym.create_box(
                    self.env.sim, ground_x, ground_y, ground_z, asset_options)
                self._asset_cache[ground_key] = (ground_asset, ground_x, ground_y, ground_z)
            else:
                ground_asset, ground_x, ground_y, ground_z = self._asset_cache[ground_key]

            # 把 box 的上表面放在 z=0 -> box 中心需下移 half height
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.01)
            pose.r = gymapi.Quat(0, 0, 0, 1)

            # 使用 _create_actor 放置，并获取 actor handle
            segmentation_id = self.env.envs_actors_num[env_idx]
            ahandle = self.env._create_actor(
                env_idx, ground_asset, 'ground', segmentation_id=segmentation_id, pose=pose)

            # 设置纯色（可见视觉和碰撞）
            try:
                color = gymapi.Vec3(0.35, 0.3, 0.25)
                self.env.gym.set_rigid_body_color(
                    self.env.envs[env_idx], ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            except Exception:
                # 部分 isaacgym 版本的签名或行为不同，忽略颜色设置失败
                pass
        except Exception as e:
            print(f"[BaseDroneEnv] failed to create ground primitive: {e}")   

        # 如果存在 placements（多方案），为当前 env 分配一个方案（按 env index 循环）并创建对应 actor
        if self.placements and len(self.placements) > 0:

            scheme = self.placements[(self._env_base_idx + env_idx) % len(self.placements)]
            # 记录已创建环境的编号
            self._created_env_idx += 1

            for p_idx, placement in enumerate(scheme):
                src = placement.get('src')
                urdf_dir, urdf_name = self._resolve_urdf_path(src)
                if urdf_dir is None or urdf_name is None:
                    continue
                
                # 获取obj_dir以及obj_name
                parent_dir, _ = os.path.split(urdf_dir)
                obj_dir = os.path.join(parent_dir, 'obj')
                obj_name = urdf_name.replace('.urdf', '.obj')

                asset_key = os.path.join(urdf_dir, urdf_name)
                if asset_key not in self._asset_cache:
                    asset_options = gymapi.AssetOptions()
                    asset_options.flip_visual_attachments = True
                    asset_options.fix_base_link = True
                    asset_options.disable_gravity = True
                    asset = self.env.gym.load_asset(
                        self.env.sim, urdf_dir, urdf_name, asset_options)
                    self._asset_cache[asset_key] = asset

                    # 尝试加载对应的 obj 作为 pytorch3d 资产（可选）
                    obj_path = os.path.join(obj_dir, obj_name)
                    self._obj_paths.append(obj_path)

                asset = self._asset_cache[asset_key]

                pos = placement.get('position', [0, 0, 0])
                yaw = placement.get('rotation', 0.0)
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(pos[0], pos[1], pos[2])
                s = math.sin(yaw / 2.0)
                c = math.cos(yaw / 2.0)
                pose.r = gymapi.Quat(0.0, 0.0, s, c)

                base_name = os.path.splitext(urdf_name)[0]
                # 保持名称在 env 间唯一
                name = f"{base_name}_env{env_idx}_{p_idx}"

                # 按类别记录 segmentation id
                class_name = parent_dir.split('/')[-1]
                if class_name not in self._class_seg_id:
                    self._class_seg_id[class_name] = self._next_class_seg_id
                    self._next_class_seg_id += 1

                # 创建资产
                try:
                    segmentation_id = self.env.envs_actors_num[env_idx]
                    # 使用当前环境已有 actor 数量作为 seg id
                    self.env._create_actor(
                        env_idx, asset, name, segmentation_id=segmentation_id, pose=pose)
                except Exception as e:
                    print(f"[BaseDroneEnv] failed to create actor {name} in env {env_idx}: {e}")

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.env.env_origins = torch.zeros(
            self.env.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.env.num_envs))
        num_rows = np.ceil(self.env.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')

        # 默认 spacing 来自配置
        spacing = float(self.config.env_spacing)

        # 如果存在 placements（多个方案），使用第一个方案的 x/y 范围来调整 spacing
        try:
            if hasattr(self, 'placements') and self.placements:
                first_scheme = self.placements[0]
                xs = [p['position'][0] for p in first_scheme if 'position' in p and len(p['position']) >= 2]
                ys = [p['position'][1] for p in first_scheme if 'position' in p and len(p['position']) >= 2]
                if xs and ys:
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)
                    # 使用配置中的 padding 来扩大 bbox，以考虑单个物体的尺寸
                    padding = getattr(self.config, 'env_padding', 10.0)
                    scene_max_dim = max(width, height)
                    scene_max_dim = scene_max_dim + 2.0 * padding
                    spacing = max(spacing, scene_max_dim)
        except Exception as e:
            print(f"[BaseDroneEnv] warning computing dynamic spacing: {e}")

        self.env.env_origins[:, 0] = spacing * xx.flatten()[:self.env.num_envs]
        self.env.env_origins[:, 1] = spacing * yy.flatten()[:self.env.num_envs]
        self.env.env_origins[:, 2] = 0.0

    def sample_assets_point_cloud(
            self, pt_per_vox = 10):
        
        meshes = load_objs_as_meshes(self._obj_paths, device=self.device)
        
        verts_list = meshes.verts_list()
        min_coords = []
        max_coords = []
        # 遍历每个网格计算边界框
        for i, verts in enumerate(verts_list):
            min_coord = verts.min(dim=0).values
            max_coord = verts.max(dim=0).values
            min_coords.append(min_coord)
            max_coords.append(max_coord)
        min_coords = torch.stack(min_coords)
        max_coords = torch.stack(max_coords)
        # 计算每个mesh的边长
        scales = max_coords - min_coords
        # 计算包围框的表面积
        areas = 2 * (scales[:, 0] * scales[:, 1] +
                     scales[:, 1] * scales[:, 2] +
                     scales[:, 0] * scales[:, 2])
        # 根据semantic map的voxel_size计算采样点数
        voxel_size = self.env.sem_voxel_size
        voxel_size = torch.min(voxel_size, dim = 0).values
        voxel_size_area = voxel_size[0] * voxel_size[1]
        num_samples = (areas / voxel_size_area * pt_per_vox).long()
        num_samples = max(num_samples)
        num_samples = max(num_samples, 60000)

        self.num_samples = num_samples
        print(f"[simple_citygen] sampling {num_samples} points per asset for semantic mapping.")

        # 从 meshes 中采样点云
        point_clouds = sample_points_from_meshes(meshes, num_samples)
        # 绕x轴旋转90度
        rotation_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ], dtype=torch.float32, device=self.device
        )
        rotation_matrixs = rotation_matrix.unsqueeze(0).repeat(
            point_clouds.shape[0], 1, 1).permute(0, 2, 1)
        point_clouds = torch.bmm(point_clouds, rotation_matrixs)

        self._asset_cache_pytorch3d = {}
        for i, obj_path in enumerate(self._obj_paths):
            self._asset_cache_pytorch3d[obj_path] = point_clouds[i]      

    def create_ground_truth_sem_map(self):
        '''
        创建场景的地面真实地图以及高度图
        需要在调用全部additional_create之后调用
        '''
        # 判断_env_base_idx是否与_created_env_idx相等
        assert self._env_base_idx + self.env.num_envs == self._created_env_idx, \
            "[simple_citygen] create_ground_truth_sem_map must be called after all environments are created."
        
        semantic_map_size = self.env.sem_map_size
        semantic_map = torch.zeros(
            self.env.num_envs,
            semantic_map_size[0], semantic_map_size[1],
            self._next_class_seg_id,
            dtype=torch.int32, device=self.device)
        env_max_coord = self.env.env_max_cord[:, 0:2]
        env_min_coord = self.env.env_min_cord[:, 0:2]
        voxel_size = self.env.sem_voxel_size

        height_map = torch.zeros(
            self.env.num_envs,
            semantic_map_size[0], semantic_map_size[1],
            dtype=torch.float32, device=self.device)

        # 如果存在 placements（多方案），为当前 env 分配一个方案（按 env index 循环）并创建对应 actor
        if self.placements and len(self.placements) > 0:

            for env_idx in tqdm(range(self.env.num_envs)):
                scheme = self.placements[(self._env_base_idx + env_idx) % len(self.placements)]

                point_clouds = []
                poses = []
                rotation_matrixs = []
                seg_idxs = []

                for p_idx, placement in enumerate(scheme):
                    src = placement.get('src')
                    urdf_dir, urdf_name = self._resolve_urdf_path(src)
                    if urdf_dir is None or urdf_name is None:
                        continue
                    
                    # 获取obj_dir以及obj_name
                    parent_dir, _ = os.path.split(urdf_dir)
                    obj_dir = os.path.join(parent_dir, 'obj')
                    obj_name = urdf_name.replace('.urdf', '.obj')
                    obj_path = os.path.join(obj_dir, obj_name)

                    # 获取点云的
                    point_cloud = self._asset_cache_pytorch3d[obj_path]
                    point_clouds.append(point_cloud)
            
                    # 构造 pose（placements.json 中的 position 与 rotation）
                    pos = placement.get('position', [0, 0, 0])
                    pos = torch.tensor(
                        pos, dtype=torch.float32, device=self.device)
                    poses.append(pos)
                    yaw = placement.get('rotation', 0.0)
                    yaw = torch.tensor(
                        yaw, dtype=torch.float32, device=self.device)
                    rotation_matrix = torch.tensor(
                        [
                            [torch.cos(yaw), -torch.sin(yaw), 0],
                            [torch.sin(yaw),  torch.cos(yaw), 0],
                            [0,             0,              1]
                        ], dtype=torch.float32, device=self.device
                    )
                    rotation_matrixs.append(rotation_matrix)

                    # 获取 segmentation id
                    class_name = parent_dir.split('/')[-1]
                    seg_idx = self._class_seg_id.get(class_name, 0)
                    seg_idxs.append(seg_idx)                   
            
                point_clouds = torch.stack(point_clouds, dim=0)
                poses = torch.stack(poses, dim=0)
                rotation_matrixs = torch.stack(
                    rotation_matrixs, dim=0).permute(0, 2, 1)
                seg_idxs = torch.tensor(seg_idxs, dtype=torch.int32, device=self.device)

                # 将seg_idxs拓展为与point_clouds对应的形状
                seg_idxs = seg_idxs.unsqueeze(1).repeat(1, point_clouds.shape[1]).view(-1)

                # 将点云应用于旋转矩阵
                point_clouds = torch.bmm(point_clouds, rotation_matrixs) + \
                    poses.unsqueeze(1)

                # 提取高度
                heights = point_clouds[:, :, 2].view(-1)                
                
                # 只保留x,y坐标
                point_clouds = point_clouds[:, :, :2].view(-1, 2)

                # 转为离散坐标
                point_cloud_idx, info = point_cloud_to_occ_idx_one_env(
                    point_clouds,
                    env_max_coord[env_idx], env_min_coord[env_idx], voxel_size[env_idx], semantic_map_size)

                if point_cloud_idx.shape[0] == 0:
                    continue
                
                # 过滤越界点
                bound_mask = info[0]
                seg_idxs = seg_idxs[bound_mask]

                # 处理不同class的点云
                for idx in range(0, self._next_class_seg_id):
                    class_mask = (seg_idxs == idx)
                    class_point_cloud_idx = point_cloud_idx[class_mask]

                    if class_point_cloud_idx.shape[0] == 0:
                        continue

                    # ensure indices are long tensors (required by PyTorch advanced indexing)
                    rows = class_point_cloud_idx[:, 0].to(torch.long)
                    cols = class_point_cloud_idx[:, 1].to(torch.long)

                    # clamp to valid map bounds as a safety measure
                    rows = torch.clamp(rows, 0, semantic_map_size[0] - 1)
                    cols = torch.clamp(cols, 0, semantic_map_size[1] - 1)

                    semantic_map[env_idx, rows, cols, idx] = 1

                # 处理高度图：在同位置聚合最大值
                heights = heights[bound_mask]

                # 使用线性索引并对具有相同格子的点取最大高度（避免重复索引赋值的不确定性）
                rows = point_cloud_idx[:, 0].to(torch.long)
                cols = point_cloud_idx[:, 1].to(torch.long)
                linear_idx = rows * semantic_map_size[1] + cols

                # 使用 torch_scatter 的 scatter_max 计算每个栅格的最大高度
                dim_size = int(semantic_map_size[0] * semantic_map_size[1])
                max_vals, _ = torch_scatter.scatter_max(heights, linear_idx, dim=0, dim_size=dim_size)

                # 将max_vals的最小值设为最小高度
                min_height = self.env.config.min_height
                max_vals = torch.clamp(max_vals, min=min_height)

                # 填回高度图
                height_map[env_idx] = max_vals.view(
                    semantic_map_size[0], semantic_map_size[1])
                height_map.cpu().numpy()
        return semantic_map, height_map