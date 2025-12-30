import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from gym import spaces

def point_cloud_to_occ_idx(pts_target, env_max_cord, env_min_cord, voxel_size, map_size):
    '''
    相对当前环境的点云，并非全局坐标
    '''
    num_env = len(pts_target)

    xyz_max_voxel = env_max_cord + 0.5 * voxel_size
    xyz_min_voxel = env_min_cord - 0.5 * voxel_size

    pts_target_idxs = []
    for env_idx in range(num_env):
        # Convert current environment points to torch tensor
        pts_env = pts_target[env_idx]
        
        # Convert to indices
        pts_target_idx = torch.floor(
            (pts_env - xyz_min_voxel[env_idx]) / voxel_size[env_idx]
        ).long()

        # Bounds checking masks
        bound_mask = (xyz_max_voxel[env_idx] > pts_env) & (pts_env > xyz_min_voxel[env_idx])
        bound_mask = torch.all(bound_mask, dim=-1)  
        # [num_pts]
        
        valid_pts = pts_target_idx[bound_mask]

        if len(valid_pts) == 0:
            pts_target_idxs.append([])
            continue

        # Unique and clip
        valid_pts = torch.unique(valid_pts, dim=0)
        min_idx = torch.zeros(
            valid_pts.shape[-1], dtype=torch.long, device=valid_pts.device)
        valid_pts = torch.clamp(valid_pts, min=min_idx, max=map_size-1)
        pts_target_idxs.append(valid_pts)

    return pts_target_idxs

def point_cloud_to_occ_idx_cnt(pts_target, env_max_cord, env_min_cord, voxel_size, map_size):
    '''
    相对当前环境的点云，并非全局坐标
    '''
    num_env = len(pts_target)

    xyz_max_voxel = env_max_cord + 0.5 * voxel_size
    xyz_min_voxel = env_min_cord - 0.5 * voxel_size

    pts_target_idxs = []
    infos = []
    for env_idx in range(num_env):

        # Convert current environment points to torch tensor
        pts_env = pts_target[env_idx]
        
        # Convert to indices
        pts_target_idx = torch.floor(
            (pts_env - xyz_min_voxel[env_idx]) / voxel_size[env_idx]
        ).long()

        # Bounds checking masks
        bound_mask = (xyz_max_voxel[env_idx] > pts_env) & (pts_env > xyz_min_voxel[env_idx])
        bound_mask = torch.all(bound_mask, dim=-1)  
        # [num_pts]
        
        valid_pts = pts_target_idx[bound_mask]

        if len(valid_pts) == 0:
            pts_target_idxs.append([])
            infos.append([])
            continue

        # Unique and clip
        valid_pts, reduced_ind, counts = torch.unique(
            valid_pts, return_inverse=True, return_counts=True,dim=0)

        # valid_pts = torch.unique(valid_pts, dim=0)
        min_idx = torch.zeros(
            valid_pts.shape[-1], dtype=torch.long, device=valid_pts.device)
        valid_pts = torch.clamp(valid_pts, min=min_idx, max=map_size-1)
        pts_target_idxs.append(valid_pts)
        info = [bound_mask, reduced_ind, counts]
        infos.append(info)

    return pts_target_idxs, infos

def point_cloud_to_occ_idx_one_env(
        pts_target, env_max_cord, env_min_cord, voxel_size, map_size):
    '''
    相对当前环境的点云，并非全局坐标
    '''
    xyz_max_voxel = env_max_cord + 0.5 * voxel_size
    xyz_min_voxel = env_min_cord - 0.5 * voxel_size

    pts_target_idx = torch.floor(
        (pts_target - xyz_min_voxel) / voxel_size
    ).long()

    # Bounds checking masks
    bound_mask = (xyz_max_voxel > pts_target) & (pts_target > xyz_min_voxel)
    bound_mask = torch.all(bound_mask, dim=-1)
    # [num_pts]

    valid_pts = pts_target_idx[bound_mask]

    if len(valid_pts) == 0:
        return []

    # # Unique and clip
    # valid_pts, reduced_ind, counts = torch.unique(valid_pts, return_inverse=True, return_counts=True, dim=0)
    # min_idx = torch.zeros(
    #     valid_pts.shape[-1], dtype=torch.long, device=valid_pts.device)
    # valid_pts = torch.clamp(valid_pts, min=min_idx, max=map_size-1)
    info = [bound_mask]

    return valid_pts, info

def pose_to_occ_idx(pose, env_min_cord, voxel_size):
    """
    Accelerated 3D version of pose coordinate to index conversion
    
    Params:
        pose: [num_step, 3], x-y-z pose
        range_gt: [num_env, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        map_size: int, size of the voxel grid (default: 256)
    
    Return:
        pose_idx: [num_env, 3]
    """
    # Calculate voxel boundaries with offset
    xyz_min_voxel = env_min_cord - 0.5 * voxel_size  
    # [num_env, 3]

    assert pose.shape[1] == 3, f"Invalid pose shape: {pose.shape}"
    pose_idx = ((pose - xyz_min_voxel) / voxel_size).floor().long()

    return pose_idx

def grid_occupancy_tri_cls(grid_prob, threshold_occu=0.5, threshold_free=0.0, return_tri_cls_only=False):
    """
    Params:
        grid_prob: [num_env, X, Y, Z], from self.grid_backproj[..., 3]

    Return:
        grid_occupancy: [num_env, X, Y, Z], voxel value among {0/1}. 0: free/unknown, 1: occupied
        grid_tri_cls: [num_env, X, Y, Z], voxel value among {-1/0/1}. -1: free, 0: unknown, 1: occupied
    """
    grid_occupancy = (grid_prob > threshold_occu).to(torch.float32)
    grid_free = (grid_prob < threshold_free).to(torch.float32)

    grid_tri_cls = grid_occupancy - grid_free   
    # element value: {-1, 0, 1}
    if return_tri_cls_only:
        return grid_tri_cls
    else:
        return grid_occupancy, grid_tri_cls

def bresenham3D_pycuda(pts_source, pts_target, map_size):
    if isinstance(map_size, list):
        assert len(map_size) == 3 and map_size[0] == map_size[1] == map_size[2], "map_size must be cubic"
        map_size = map_size[0]

    # Keep data on GPU if already there
    device = pts_source.device
    source_pts = pts_source.int().contiguous()
    target_pts = pts_target.int().contiguous()
    num_rays = target_pts.shape[0]
    
    # Optimize max_pts_per_ray calculation based on manhattan distance
    # max_pts_per_ray = min(map_size * 3, int(1.2 * torch.max(torch.abs(target_pts - source_pts.expand_as(target_pts)).sum(dim=1))))
    max_pts_per_ray = map_size * 3
    
    # Allocate output memory directly on GPU
    trajectory_pts = torch.zeros((num_rays, max_pts_per_ray, 3), dtype=torch.int32, device=device)
    trajectory_lengths = torch.zeros(num_rays, dtype=torch.int32, device=device)

    kernel_code = """
    __device__ __forceinline__ int max3(int a, int b, int c) {
        return max(max(a, b), c);
    }

    __device__ void bresenham_line_3d(
        const int x0, const int y0, const int z0,
        const int x1, const int y1, const int z1,
        int *__restrict__ trajectory,
        int *__restrict__ length,
        const int map_size,
        const int max_pts_per_ray) {
        
        const int dx = abs(x1 - x0);
        const int dy = abs(y1 - y0);
        const int dz = abs(z1 - z0);
        
        const int sx = (x0 < x1) ? 1 : -1;
        const int sy = (y0 < y1) ? 1 : -1;
        const int sz = (z0 < z1) ? 1 : -1;
        
        const int dm = max3(dx, dy, dz);
        int x = x0, y = y0, z = z0;
        int idx = 0;
        
        #pragma unroll 1
        if (dm == dx) {
            int p1 = 2 * dy - dx;
            int p2 = 2 * dz - dx;
            
            // Pre-compute bounds check
            const bool x_valid = (x >= 0 && x < map_size);
            const bool y_valid = (y >= 0 && y < map_size);
            const bool z_valid = (z >= 0 && z < map_size);
            
            if (x_valid && y_valid && z_valid) {
                trajectory[idx * 3] = x;
                trajectory[idx * 3 + 1] = y;
                trajectory[idx * 3 + 2] = z;
                idx++;
            }
            
            #pragma unroll 4
            for (int i = 0; i < dx && idx < max_pts_per_ray; i++) {
                if (p1 >= 0) { y += sy; p1 -= 2 * dx; }
                if (p2 >= 0) { z += sz; p2 -= 2 * dx; }
                x += sx;
                p1 += 2 * dy;
                p2 += 2 * dz;
                
                if (x >= 0 && x < map_size && 
                    y >= 0 && y < map_size && 
                    z >= 0 && z < map_size) {
                    trajectory[idx * 3] = x;
                    trajectory[idx * 3 + 1] = y;
                    trajectory[idx * 3 + 2] = z;
                    idx++;
                }
            }
        } else if (dm == dy) {
            // Similar optimizations for dy dominant case
            int p1 = 2 * dx - dy;
            int p2 = 2 * dz - dy;
            
            if (x >= 0 && x < map_size && 
                y >= 0 && y < map_size && 
                z >= 0 && z < map_size) {
                trajectory[idx * 3] = x;
                trajectory[idx * 3 + 1] = y;
                trajectory[idx * 3 + 2] = z;
                idx++;
            }
            
            #pragma unroll 4
            for (int i = 0; i < dy && idx < max_pts_per_ray; i++) {
                if (p1 >= 0) { x += sx; p1 -= 2 * dy; }
                if (p2 >= 0) { z += sz; p2 -= 2 * dy; }
                y += sy;
                p1 += 2 * dx;
                p2 += 2 * dz;
                
                if (x >= 0 && x < map_size && 
                    y >= 0 && y < map_size && 
                    z >= 0 && z < map_size) {
                    trajectory[idx * 3] = x;
                    trajectory[idx * 3 + 1] = y;
                    trajectory[idx * 3 + 2] = z;
                    idx++;
                }
            }
        } else {
            // Similar optimizations for dz dominant case
            int p1 = 2 * dx - dz;
            int p2 = 2 * dy - dz;
            
            if (x >= 0 && x < map_size && 
                y >= 0 && y < map_size && 
                z >= 0 && z < map_size) {
                trajectory[idx * 3] = x;
                trajectory[idx * 3 + 1] = y;
                trajectory[idx * 3 + 2] = z;
                idx++;
            }
            
            #pragma unroll 4
            for (int i = 0; i < dz && idx < max_pts_per_ray; i++) {
                if (p1 >= 0) { x += sx; p1 -= 2 * dz; }
                if (p2 >= 0) { y += sy; p2 -= 2 * dz; }
                z += sz;
                p1 += 2 * dx;
                p2 += 2 * dy;
                
                if (x >= 0 && x < map_size && 
                    y >= 0 && y < map_size && 
                    z >= 0 && z < map_size) {
                    trajectory[idx * 3] = x;
                    trajectory[idx * 3 + 1] = y;
                    trajectory[idx * 3 + 2] = z;
                    idx++;
                }
            }
        }
        
        *length = idx;
    }

    __global__ void ray_casting_kernel_3d(
        const int *__restrict__ source_pts,
        const int *__restrict__ target_pts,
        int *__restrict__ trajectory_pts,
        int *__restrict__ trajectory_lengths,
        const int num_rays,
        const int map_size,
        const int max_pts_per_ray) {
        
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= num_rays) return;
        
        const int src_x = source_pts[0];
        const int src_y = source_pts[1];
        const int src_z = source_pts[2];
        const int tgt_x = target_pts[ray_idx * 3];
        const int tgt_y = target_pts[ray_idx * 3 + 1];
        const int tgt_z = target_pts[ray_idx * 3 + 2];
        
        bresenham_line_3d(
            src_x, src_y, src_z,
            tgt_x, tgt_y, tgt_z,
            &trajectory_pts[ray_idx * max_pts_per_ray * 3],
            &trajectory_lengths[ray_idx],
            map_size,
            max_pts_per_ray
        );
    }
    """
    
    # Compile kernel with optimization flags
    mod = SourceModule(kernel_code, options=['-O3'])
    ray_casting_kernel = mod.get_function("ray_casting_kernel_3d")
    
    # Configure kernel launch parameters
    block_size = 256
    grid_size = (num_rays + block_size - 1) // block_size
    
    # Launch kernel with streamed execution
    stream = cuda.Stream()
    ray_casting_kernel(
        source_pts,
        target_pts,
        trajectory_pts,
        trajectory_lengths,
        np.int32(num_rays),
        np.int32(map_size),
        np.int32(max_pts_per_ray),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
        stream=stream
    )
    
    # Process results efficiently using GPU operations
    mask = torch.arange(max_pts_per_ray, device=device)[None, :] < trajectory_lengths[:, None]
    mask = mask.unsqueeze(-1).expand(-1, -1, 3)
    results = trajectory_pts[mask].view(-1, 3)

    return results.to(torch.long)

def bresenham_2d(pts_source, pts_target, map_size):
    """
    2D Bresenham line algorithm implemented by PyCUDA for GPU acceleration.
    """
    if isinstance(map_size, list):
        assert len(map_size) == 2 and map_size[0] == map_size[1], "map_size must be a square"
        map_size = map_size[0]

    # Keep data on GPU if already there
    device = pts_source.device
    source_pts = pts_source.int().contiguous().to(device)
    target_pts = pts_target.int().contiguous().to(device)
    num_rays = target_pts.shape[0]
    
    # Optimize max_pts_per_ray calculation based on manhattan distance
    # max_pts_per_ray = min(map_size * 2, 
    #     int(1.2 * torch.max(torch.abs(target_pts - source_pts.expand_as(target_pts)).sum(dim=1))))
    max_pts_per_ray = map_size * 2

    # Allocate output memory directly on GPU
    trajectory_pts = torch.zeros((num_rays, max_pts_per_ray, 2), dtype=torch.int32, device=device)
    trajectory_lengths = torch.zeros(num_rays, dtype=torch.int32, device=device)

    kernel_code = """
    __device__ __forceinline__ void bresenham_line(
        const int x0, const int y0,
        const int x1, const int y1,
        int *__restrict__ trajectory,
        int *__restrict__ length,
        const int map_size,
        const int max_pts_per_ray) {
        
        const int dx = abs(x1 - x0);
        const int dy = abs(y1 - y0);
        const int sx = (x0 < x1) ? 1 : -1;
        const int sy = (y0 < y1) ? 1 : -1;
        
        int x = x0;
        int y = y0;
        int err = dx - dy;
        int idx = 0;
        
        // Pre-compute bounds check for first point
        const bool initial_valid = (x >= 0 && x < map_size && y >= 0 && y < map_size);
        if (initial_valid) {
            trajectory[0] = x;
            trajectory[1] = y;
            idx = 1;
        }
        
        #pragma unroll 4
        while (idx < max_pts_per_ray) {
            if (x == x1 && y == y1) break;
            
            const int e2 = 2 * err;
            
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
            
            if (x >= 0 && x < map_size && y >= 0 && y < map_size) {
                trajectory[idx * 2] = x;
                trajectory[idx * 2 + 1] = y;
                idx++;
            }
        }
        
        *length = idx;
    }

    __global__ void ray_casting_kernel(
        const int *__restrict__ source_pts,
        const int *__restrict__ target_pts,
        int *__restrict__ trajectory_pts,
        int *__restrict__ trajectory_lengths,
        const int num_rays,
        const int map_size,
        const int max_pts_per_ray) {
        
        const int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (ray_idx >= num_rays) return;
        
        const int src_x = source_pts[0];
        const int src_y = source_pts[1];
        const int tgt_x = target_pts[ray_idx * 2];
        const int tgt_y = target_pts[ray_idx * 2 + 1];
        
        bresenham_line(
            src_x, src_y,
            tgt_x, tgt_y,
            &trajectory_pts[ray_idx * max_pts_per_ray * 2],
            &trajectory_lengths[ray_idx],
            map_size,
            max_pts_per_ray
        );
    }
    """
    
    # Compile kernel with optimization flags
    mod = SourceModule(kernel_code, options=['-O3'])
    ray_casting_kernel = mod.get_function("ray_casting_kernel")
    
    # Configure kernel launch parameters
    block_size = 256
    grid_size = (num_rays + block_size - 1) // block_size
    
    # Launch kernel with streamed execution
    stream = cuda.Stream()
    ray_casting_kernel(
        source_pts,
        target_pts,
        trajectory_pts,
        trajectory_lengths,
        np.int32(num_rays),
        np.int32(map_size),
        np.int32(max_pts_per_ray),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
        stream=stream
    )
    
    # Process results efficiently using GPU operations
    mask = torch.arange(max_pts_per_ray, device=device)[None, :] < trajectory_lengths[:, None]
    mask = mask.unsqueeze(-1).expand(-1, -1, 2)
    results = trajectory_pts[mask].view(-1, 2)

    return results.to(torch.long)

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def scanned_pts_to_idx_3D(pts_target, range_gt, voxel_size_gt, map_size=256):
    """
    Params:
        pts_target: [num_env, num_pts, 3], list of torch.tensor, target points by back-projection
        range_gt_scenes: [num_env, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        voxel_size_scenes: [num_env, 3]

    Return:
        pts_target_idxs: list of (num_valid_pts_idx, 3)
    """
    num_env = len(pts_target)

    xyz_max_voxel = range_gt[:, [0,2,4]] + 0.5 * voxel_size_gt
    xyz_min_voxel = range_gt[:, [1,3,5]] - 0.5 * voxel_size_gt

    pts_target_idxs = []
    for env_idx in range(num_env):
        # Convert current environment points to torch tensor
        pts_env = pts_target[env_idx]
        
        # Convert to indices
        pts_target_idx = torch.floor(
            (pts_env - xyz_min_voxel[env_idx]) / voxel_size_gt[env_idx]
        ).long()

        # Bounds checking masks
        bound_mask = (xyz_max_voxel[env_idx] > pts_env) & (pts_env > xyz_min_voxel[env_idx])
        bound_mask = torch.all(bound_mask, dim=-1)  # [num_pts]

        valid_pts = pts_target_idx[bound_mask]

        if len(valid_pts) == 0:
            pts_target_idxs.append([])
            continue

        # Unique and clip
        valid_pts = torch.unique(valid_pts, dim=0)
        valid_pts = torch.clamp(valid_pts, min=0, max=map_size-1)
        pts_target_idxs.append(valid_pts)

    return pts_target_idxs


def pose_coord_to_idx_3D(poses, range_gt, voxel_size_gt, map_size=256, if_col=False):
    """
    Accelerated 3D version of pose coordinate to index conversion
    
    Params:
        poses: [num_step, 3], x-y-z pose
        range_gt: [num_env, 6], (x_max, x_min, y_max, y_min, z_max, z_min) of N_gt
        map_size: int, size of the voxel grid (default: 256)
    
    Return:
        poses_idx: [num_env, 3]
    """
    # Extract minimum bounds for each dimension
    x_min = range_gt[:, 1]  # [num_env]
    y_min = range_gt[:, 3]  # [num_env]
    z_min = range_gt[:, 5]  # [num_env]
    
    # Stack minimum bounds
    xyz_min = torch.stack([x_min, y_min, z_min], dim=-1)  # [num_env, 3]
    
    # Calculate voxel boundaries with offset
    xyz_min_voxel = xyz_min - 0.5 * voxel_size_gt  # [num_env, 3]
    
    assert poses.shape[1] == 3, f"Invalid poses shape: {poses.shape}"
    poses_idx = ((poses - xyz_min_voxel) / voxel_size_gt).floor().long()

    # if not if_col:
    #     # for computing ray casting, clip values to valid range
    #     poses_idx = torch.clip(poses_idx, min=0, max=map_size-1)
    if if_col:
        # for collision check
        poses_idx[(poses_idx < 0).any(dim=-1)] = -1
        poses_idx[(poses_idx > map_size-1).any(dim=-1)] = -1
    return poses_idx

def flatten_observations(observation_dict, key_sequence):
    """Flattens the observation dictionary to an array.

    If observation_excluded is passed in, it will still return a dictionary,
    which includes all the (key, observation_dict[key]) in observation_excluded,
    and ('other': the flattened array).

    Args:
      observation_dict: A dictionary of all the observations.
      key_sequence: A list/tuple of all the keys of the observations to be
        added during flattening.

    Returns:
      An array or a dictionary of observations based on whether
        observation_excluded is empty.
    """
    observations = []
    for key in key_sequence:
        value = observation_dict[key]
        num_env = value.shape[0]
        # assert key in ['state', 'state_rgb', 'grid']
        # if key == "grid":
        #     value = value.reshape(num_env, -1)    # [num_envs, 4, X, Y, Z] -> [num_envs, 4 * X * Y * Z]
        # elif key == 'state_rgb':
        #     value = value.reshape(num_env, -1)   # [num_env, k, H, W] -> [num_env, k * H * W]
        value = value.reshape(num_env, -1)
        observations.append(value)

    flat_observations = torch.concat(observations, dim=-1)  # grid observation
    return flat_observations


def flatten_observation_spaces(observation_spaces, key_sequence):
    """Flattens the dictionary observation spaces to gym.spaces.Box.

    If observation_excluded is passed in, it will still return a dictionary,
    which includes all the (key, observation_spaces[key]) in observation_excluded,
    and ('other': the flattened Box space).

    Args:
      observation_spaces: A dictionary of all the observation spaces.
      key_sequence: A list/tuple of all the keys of the observations to be
        added during flattening.

    Returns:
      A box space or a dictionary of observation spaces based on whether
        observation_excluded is empty.
    """
    assert isinstance(key_sequence, list)
    lower_bound = []
    upper_bound = []
    for key in key_sequence:
        value = observation_spaces.spaces[key]
        if isinstance(value, spaces.Box):
            lower_bound.append(np.asarray(value.low).flatten())
            upper_bound.append(np.asarray(value.high).flatten())

    lower_bound = np.concatenate(lower_bound)
    upper_bound = np.concatenate(upper_bound)
    observation_space = spaces.Box(np.array(lower_bound, dtype=np.float32), np.array(upper_bound, dtype=np.float32), dtype=np.float32)
    return observation_space