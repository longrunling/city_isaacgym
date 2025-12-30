from Envs.RLDrone.BaseDroneEnv import BaseDroneEnv
from Configs.RLDrone.Custom.Config import OccMapDrone_Config
import torch
import numpy as np
import open3d as o3d
from Envs.utils import *
from isaacgym import gymtorch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import os

class OccMapDrone(BaseDroneEnv):

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

        self.config = OccMapDrone_Config

        # 创建缓冲区
        self._create_buffers_Occ()

        # 计算gt
        # self.create_gt()
        # self.create_gt_gennbv()

    def _create_buffers_Occ(self):

        # NOTE: only compute once
        self.blender2opencv = torch.FloatTensor([[1, 0, 0, 0],
                                                [0, -1, 0, 0],
                                                [0, 0, -1, 0],
                                                [0, 0, 0, 1]]).to(self.device)
        
        # 点云下采样步长
        downsample_factor = self.config.downsample_factor
        H, W = self.config.Camera.height, self.config.Camera.width

        # 像素坐标初始化
        xs = torch.linspace(0, W-downsample_factor, int(W/downsample_factor), 
                            dtype=torch.float32, device=self.device)
        ys = torch.linspace(0, H-downsample_factor, int(H/downsample_factor), 
                            dtype=torch.float32, device=self.device)
        ys, xs = torch.meshgrid(ys, xs, indexing='ij')
        norm_coord_pixel = torch.stack([xs, ys], dim=-1)    
        # [H, W, 2]
        self.norm_coord_pixel = torch.concat((
            norm_coord_pixel, 
            torch.ones_like(norm_coord_pixel[..., :1], device=self.device)), 
            dim=-1).view(-1, 3)  
        # [H*W, 3], (u, v, 1)

        # 相机内参矩阵及其逆
        intrinsics = self.get_camera_intrinsics()   
        # [3, 3]
        self.inv_intri = torch.linalg.inv(intrinsics).to(self.device).to(torch.float32) 

        # 初始化occ_map
        occ_map_size = self.config.occ_map_size
        self.occ_map_size = torch.tensor(
            occ_map_size, dtype=torch.int32, device=self.device)
        self.occ_map = torch.zeros(
            self.num_envs, occ_map_size[0], occ_map_size[1], occ_map_size[2],
            dtype=torch.float32, device=self.device, requires_grad=False)
        # 用于计算reward的gt占用
        self.scanned_gt_occ = torch.zeros_like(self.occ_map)
        # 分类后的占用地图
        self.occ_map_cls = torch.zeros_like(self.occ_map)
        
        # occ_map的体素尺寸
        voxel_size = (self.env_max_cord - self.env_min_cord) / self.occ_map_size
        self.occ_voxel_size = voxel_size
        
    def get_camera_intrinsics(self):

        H, W = self.config.Camera.height, self.config.Camera.width
        FOV_x = self.config.Camera.horizontal_fov / 180 * np.pi
        FOV_y = FOV_x * H / W   
        # Vertical field of view is calculated from height to width ratio

        focal_x = 0.5 * W / np.tan(0.5 * FOV_x)
        focal_y = 0.5 * H / np.tan(0.5 * FOV_y)
        cx, cy = W / 2, H / 2
        intrinsics = torch.tensor([[focal_x, 0, cx], [0, focal_y, cy], [0, 0, 1]]).float()
        return intrinsics           

    def get_camera_view_matrix(self):
        """
        return Extrinsics.t() instead of Extrinsics. E * P = P * E.t()
        """
        #  We assume the number of envs equals to th number of cameras"
        assert len(self.cameras) == self.num_envs
        ret = []
        for k, handle in enumerate(self.cameras):
            # camera_pose = self.gym.get_camera_transform(self.envs[k], handle)
            ret.append(self.gym.get_camera_view_matrix(self.sim, self.envs[k], handle))
            # env_origin = self.gym.get_env_origin(self.envs[k])
        return np.array(ret)
    
    def get_seg_id_mask(self, min_seg_id = 2):
        '''
        获取前景掩码,seg_id >= min_seg_id
        '''
        seg_maps = self.seg_processed.clone()
        seg_mask = (seg_maps >= min_seg_id)
        return seg_mask.reshape(self.num_envs, -1)

    def get_point_cloud(self, min_seg_id = 2):
        '''
        1为地面
        coords_world存储前景点云,fg_coord存储前景像素坐标
        '''
        downsample_factor = self.config.downsample_factor
        depth_maps = self.depth_processed.clone()
        rgb_maps = self.rgb_processed.clone().permute(0, 2, 3, 1)
        # [num_env, H, W]
        depth_maps_fg = (self.seg_processed.clone() >= min_seg_id)
        if downsample_factor != 1:
            depth_maps = depth_maps[:, ::downsample_factor, ::downsample_factor]   
            # [num_env, H_down, W_down]
            depth_maps_fg = depth_maps_fg[:, ::downsample_factor, ::downsample_factor]
            rgb_maps = rgb_maps[:, ::downsample_factor, ::downsample_factor, :]

        depth_maps[~depth_maps_fg] = 0.

        # 提取depth为0处的坐标，稍后用于去除多余点云
        non_zero_mask = ~(depth_maps == 0.)
        depth_maps_fg = depth_maps_fg & non_zero_mask
        # [num_env, H*W]

        # NOTE: back-projection
        extrinsics = torch.from_numpy(self.get_camera_view_matrix()).to(self.device) 
        # [num_env, 4, 4]
        c2w = torch.linalg.inv(extrinsics.transpose(-2, -1)) @ self.blender2opencv.unsqueeze(0)
        c2w[:, :3, 3] -= self.env_origins

        # num_point == H * W
        depth_maps = depth_maps.reshape(self.num_envs, -1)          
        # [num_env, H*W]
        depth_maps_fg = depth_maps_fg.reshape(self.num_envs, -1)    
        # [num_env, H*W]
        rgb_maps = rgb_maps.reshape(self.num_envs, -1, 3)
        # [num_env, H*W, 3]
        coords_pixel = torch.einsum('ij,jk->ijk', depth_maps, self.norm_coord_pixel)   
        # [num_env, num_point, 3]

        # inv_intri: [3, 3], coords_pixel: [num_env, num_point, 3]
        coords_cam = torch.einsum('ij,nkj->nki', self.inv_intri, coords_pixel)    
        # [num_env, num_point, 3]
        coords_cam_homo = torch.concat((coords_cam, torch.ones_like(coords_cam[..., :1], device=self.device)), dim=-1)   
        # [num_env, num_point, 4], homogeneous format

        # c2w: [num_env, 4, 4], coord_cam_homo: [num_env, num_point, 4]
        coords_world = torch.einsum('nij,nkj->nki', c2w, coords_cam_homo)[..., :3]
        coords_world_rgb = rgb_maps
        # [num_env, num_point, 4] -> [num_env, num_point, 3]

        # 使用non_zero_mask进行过滤
        new_coords_world = []
        new_coords_world_rgb = []
        for idx in range(self.num_envs):
            temp_coords = coords_world[idx][depth_maps_fg[idx]]
            temp_rgb = coords_world_rgb[idx][depth_maps_fg[idx], :]
            new_coords_world.append(temp_coords)
            new_coords_world_rgb.append(temp_rgb)

        self.coords_world = new_coords_world
        self.coords_world_rgb = new_coords_world_rgb
        self.fg_coord = depth_maps_fg
        # list of [num_points, 3]

    def update_occ_map(self, min_seg_id=1):

        """ Update scanned probabilistic grids. """
        self.get_point_cloud(min_seg_id)   
        pts_target = self.coords_world
        # list of target points, num_env * [n, 3]

        # num_layer lists of (num_valid_pts_idx, 3)
        # env_max_cord = self.range_gt[:, 0::2]
        # env_min_cord = self.range_gt[:, 1::2]
        env_max_cord = self.env_max_cord
        env_min_cord = self.env_min_cord
        # map_size = torch.ones_like(self.occ_map_size) * self.grid_size
        map_size = self.occ_map_size
        pts_idx_all = point_cloud_to_occ_idx(
            pts_target,
            env_max_cord,
            env_min_cord,
            # self.voxel_size_gt,
            self.occ_voxel_size,
            map_size)
        
        pose_idx_3D = pose_to_occ_idx(
            self.pose[:, 0:3],
            env_min_cord,
            self.occ_voxel_size)
        
        occ_grids = torch.zeros(
            self.num_envs, 
            self.occ_map_size[0], self.occ_map_size[1], self.occ_map_size[2],
            dtype=torch.float32, device=self.device)

        for env_idx in range(self.num_envs):
            pts_idx_3D = pts_idx_all[env_idx]   
            # [num_point, 3]

            if (isinstance(pts_idx_3D, list) and len(pts_idx_3D) == 0) or \
                pts_idx_3D.shape[0] == 0:
                continue

            occ_grids[env_idx, pts_idx_3D[:, 0], pts_idx_3D[:, 1], pts_idx_3D[:, 2]] = 1.0

            # [num_point, 3] for representation
            ray_cast_paths_3D = bresenham3D_pycuda(
                pts_source=pose_idx_3D[env_idx: env_idx+1],
                pts_target=pts_idx_3D,
                map_size=max(self.occ_map_size.tolist()))
            # clip the rays
            min_idx = torch.zeros(
                ray_cast_paths_3D.shape[-1], dtype=torch.long, device=self.device)
            ray_cast_paths_3D = torch.clamp(ray_cast_paths_3D, min=min_idx, max=map_size-1)

            # update
            self.occ_map[
                env_idx, 
                ray_cast_paths_3D[:, 0],
                ray_cast_paths_3D[:, 1], 
                ray_cast_paths_3D[:, 2]] -= 0.05
            self.occ_map[
                env_idx, 
                pts_idx_3D[:, 0],
                pts_idx_3D[:, 1], 
                pts_idx_3D[:, 2]] = 1.0

        # 使用阈值确定的occ_map, -1: free, 0: unknown, 1: occupied
        self.occ_map_cls = grid_occupancy_tri_cls(
            self.occ_map,
            threshold_occu=0.5,
            threshold_free=0.0,
            return_tri_cls_only=True)
        
        # # NOTE: reward computation
        # self.scanned_gt_occ = torch.clip(
        #     self.scanned_gt_occ + occ_grids * self.occ_map_gt,
        #     max=1, min=0
        # )                
            
    def reset_idx(self, env_idx):
        '''
        重置env_idx指定的环境
        '''
        num_envs = len(env_idx)
        if num_envs == 0:
            return

        # 必须调用父类的reset_envs
        super().reset_idx(env_idx)

        # 重置occ_map
        self.occ_map[env_idx] = 0
        self.scanned_gt_occ[env_idx] = 0

    def sample_pose_idx(self):

        min_tensor = self.pose_low_idx
        max_tensor = self.pose_up_idx
                
        # 生成 [0, 1) 的随机 Tensor
        random_uniform = torch.rand(self.num_envs, 6).to(self.device)
        
        # 线性插值到 [min, max] 范围
        random_pose_idx = (min_tensor + random_uniform * (max_tensor - min_tensor)).long()
        return random_pose_idx
    
    def create_gt_gennbv(self):
        """
        load ground truth data for training.
        """
        # [num_scene, X, Y, Z, 4]
        grid_gt = torch.load(
            'Datas/gt/train_houses3k_grid_gt.pt',
            map_location=self.device
        )
        self.num_scene = grid_gt.shape[0]

        # [num_scene, 3]
        self.voxel_size_gt = torch.cat(
            [grid_gt[:, 1, 0, 0, 0:1] - grid_gt[:, 0, 0, 0, 0:1],
             grid_gt[:, 0, 1, 0, 1:2] - grid_gt[:, 0, 0, 0, 1:2],
             grid_gt[:, 0, 0, 1, 2:3] - grid_gt[:, 0, 0, 0, 2:3]], dim=-1)

        # [num_scene]
        self.num_valid_voxels_gt = grid_gt[..., 3].sum(dim=(-1, -2, -3))

        # [num_scene, 6], (x_max, x_min, y_max, y_min, z_max, z_min)
        x_range = grid_gt[:, -1, 0, 0, 0:1] - grid_gt[:, 0, 0, 0, 0:1]
        y_range = grid_gt[:, 0, -1, 0, 1:2] - grid_gt[:, 0, 0, 0, 1:2]
        z_range = grid_gt[:, 0, 0, -1, 2:3] - grid_gt[:, 0, 0, 0, 2:3]
        self.range_gt = torch.cat(
            [x_range / 2, -x_range / 2,
             y_range / 2, -y_range / 2,
             z_range, torch.zeros_like(z_range)], dim=-1)

        # [X, Y, Z]
        self.grid_size = grid_gt.shape[1]
        assert grid_gt.shape[1] == grid_gt.shape[2] == grid_gt.shape[3]

        # num_scene (scenes from dataset) -> num_env (training env)
        self.env_to_scene = []
        for env_idx in range(self.num_envs):
            self.env_to_scene.append(env_idx % self.num_scene)
        self.env_to_scene = torch.tensor(self.env_to_scene, device=self.device)     
        # [num_env]

        self.occ_map_gt = grid_gt[..., 3] 
        # [num_scene, X, Y, Z, 4] -> [num_scene, X, Y, Z]
        self.occ_map_gt = self.occ_map_gt[self.env_to_scene]
        self.num_valid_voxels_gt = self.num_valid_voxels_gt[self.env_to_scene]
        self.range_gt = self.range_gt[self.env_to_scene]
        self.voxel_size_gt = self.voxel_size_gt[self.env_to_scene]

    def visualize_point_cloud(self, env_idx):

        vis_pcd = self.coords_world[env_idx].cpu().numpy()
        vis_pcd_color = self.coords_world_rgb[env_idx].cpu().numpy() / 255.0

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5,  # 坐标轴长度
            origin=[0, 0, 0]  # 坐标系原点
        )        

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_pcd)
        pcd.colors = o3d.utility.Vector3dVector(vis_pcd_color)
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

    def visualize_occ_map(self, env_idx):

        occupancy = self.occ_map[env_idx].cpu().numpy()  # 获取指定环境的占用数据
        # occupancy = self.occ_map_gt[env_idx].cpu().numpy()  # 获取指定环境的占用数据
        # voxel_size = self.voxel_size[env_idx].cpu().numpy()  # 获取体素大小（假设各轴体素大小相同）

        # 3. 创建体素网格
        # 将占用数据转换为体素坐标（仅保留占用的体素）
        occupied = np.array(np.where(occupancy > 0)).T
        unknown = np.array(np.where(occupancy == 0)).T

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.occ_map_size[0],  # 坐标轴长度
            origin=[0, 0, 0]  # 坐标系原点
        )        

        # 创建点云（体素中心点）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(occupied)
        # 为点云添加颜色（红色表示占用）
        colors = np.ones((len(occupied), 3)) * [1, 0, 0]  # RGB红色
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(unknown)
        # 为点云添加颜色（灰色表示未知）
        colors2 = np.ones((len(unknown), 3)) * [0.5, 0.5, 0.5]  # RGB灰色
        pcd2.colors = o3d.utility.Vector3dVector(colors2)

        # o3d.visualization.draw_geometries([pcd, pcd2])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

    # for gnnbv
    def back_projection_fg(self, downsample_factor=1, visualize=False):
        """ Back-projection of foreground depth maps to 3D points in world coordinate
        Args:
            downsample_factor: downsample factor
            visualize: if True, also return colors_world
        Returns:
            coords_world: list of [num_points, 3]
            colors_world: list of [num_points, 3] (if visualize)
        """
        depth_maps = self.depth_processed.clone()   # [num_env, H, W]
        depth_maps_fg = (self.seg_processed.clone() > 50)
        if downsample_factor != 1:
            depth_maps = depth_maps[:, ::downsample_factor, ::downsample_factor]   # [num_env, H_down, W_down]
            depth_maps_fg = depth_maps_fg[:, ::downsample_factor, ::downsample_factor]

        depth_maps[~depth_maps_fg] = 0.

        # NOTE: back-projection
        extrinsics = torch.from_numpy(self.get_camera_view_matrix()).to(self.device) # [num_env, 4, 4]
        c2w = torch.linalg.inv(extrinsics.transpose(-2, -1)) @ self.blender2opencv.unsqueeze(0)
        c2w[:, :3, 3] -= self.env_origins

        # num_point == H * W
        depth_maps = depth_maps.reshape(self.num_envs, -1)          # [num_env, H*W]
        depth_maps_fg = depth_maps_fg.reshape(self.num_envs, -1)    # [num_env, H*W]
        coords_pixel = torch.einsum('ij,jk->ijk', depth_maps, self.norm_coord_pixel)   # [num_env, num_point, 3]

        # inv_intri: [3, 3], coords_pixel: [num_env, num_point, 3]
        coords_cam = torch.einsum('ij,nkj->nki', self.inv_intri, coords_pixel)    # [num_env, num_point, 3]
        coords_cam_homo = torch.concat((coords_cam, torch.ones_like(coords_cam[..., :1], device=self.device)), dim=-1)   # [num_env, num_point, 4], homogeneous format

        # c2w: [num_env, 4, 4], coord_cam_homo: [num_env, num_point, 4]
        coords_world = torch.einsum('nij,nkj->nki', c2w, coords_cam_homo)[..., :3]    # [num_env, num_point, 4] -> [num_env, num_point, 3]
        coords_world = [coords_world[idx][depth_maps_fg[idx, :]] for idx in range(self.num_envs)]

        return coords_world     
        # list of [num_points, 3]