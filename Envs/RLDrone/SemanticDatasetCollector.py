from Envs.RLDrone.SemanticOccDrone import SemanticOccDrone
from Configs.RLDrone.Custom.Config import SemanticDataset_Config
import torch
import torch_scatter
import os
import math
from datetime import datetime
from Envs.utils import point_cloud_to_occ_idx_cnt

class SemanticDatasetCollector(SemanticOccDrone):
    def __init__(self, VecTask_cfg, rl_device, sim_device, graphics_device_id, headless, **kwargs):
        super().__init__(VecTask_cfg, rl_device, sim_device, graphics_device_id, headless, **kwargs)
        self.dataset_cfg = SemanticDataset_Config
        self._init_dataset_buffers()
        self._init_class_maps()
        self.dataset_index = []
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join('runs', ts, 'dataset')
        os.makedirs(self.run_dir, exist_ok=True)
        h = self.sem_map_size[0].item()
        w = self.sem_map_size[1].item()
        self.stats = {'uniform_count': 0, 'onehot_count': 0, 'total_voxels': int(self.num_envs * h * w), 'steps': 0}

    def _init_dataset_buffers(self):
        c = len(self.texts)
        h = self.sem_map_size[0].item()
        w = self.sem_map_size[1].item()
        self.votes = torch.zeros(self.num_envs, c, h, w, dtype=torch.float32, device=self.device)

    def _init_class_maps(self):
        self.seg_id_to_cls = getattr(self.simple_citygen, '_seg_id_to_cls', {})

    def _nearby_sample_pose(self, step_idx):
        b = self.simple_citygen.boundaries
        low_limit = torch.tensor([b[0], b[2], self.dataset_cfg.height_low], device=self.device)
        high_limit = torch.tensor([b[1], b[3], self.dataset_cfg.height_high], device=self.device)
        fov_rad = math.radians(self.dataset_cfg.Camera.horizontal_fov)
        valid_found = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        final_pose = self.pose.clone()
        for attempt in range(self.dataset_cfg.max_attempts):
            if valid_found.all():
                break
            if step_idx == 0:
                test_pos = torch.rand((self.num_envs, 3), device=self.device) * (high_limit - low_limit) + low_limit
            else:
                current_h = self.pose[:, 2]
                max_move = current_h * math.tan(fov_rad / 2) * 0.9
                offset = (torch.rand((self.num_envs, 3), device=self.device) * 2 - 1)
                offset[:, 0:2] *= max_move.unsqueeze(-1)
                offset[:, 2] *= self.dataset_cfg.height_change_scale
                test_pos = torch.clamp(self.pose[:, :3] + offset, low_limit, high_limit)
            test_pitch = (torch.pi / 2) + (torch.rand(self.num_envs, device=self.device) - 0.5) * (math.radians(self.dataset_cfg.pitch_range_deg))
            test_yaw = torch.rand(self.num_envs, device=self.device) * 2 * torch.pi
            test_rot = torch.stack([torch.zeros_like(test_yaw), test_pitch, test_yaw], dim=-1)
            test_pose = torch.cat([test_pos, test_rot], dim=-1)
            self.update_pose(test_pose)
            self.update_visual()
            min_depth, _ = torch.min(self.depth_processed.view(self.num_envs, -1), dim=-1)
            max_depth, _ = torch.max(self.depth_processed.view(self.num_envs, -1), dim=-1)
            is_valid = (min_depth > 0.5) & (max_depth > 3.0)
            mask = is_valid & (~valid_found)
            final_pose[mask] = test_pose[mask]
            valid_found[mask] = True
        self.update_pose(final_pose)
        self.update_visual()

    def _update_world_fusion(self):
        self.update_semantic_occ_map()

    def _accumulate_gt_votes(self):
        self.get_point_cloud(min_seg_id=0)
        env_max_cord = self.env_max_cord[:, :2]
        env_min_cord = self.env_min_cord[:, :2]
        h = self.sem_map_size[0].item()
        w = self.sem_map_size[1].item()
        for env_idx in range(self.num_envs):
            seg_img = self.seg_processed[env_idx]
            coords = self.coords_world[env_idx]
            if coords.shape[0] == 0:
                continue
            idx_valid = self.fg_coord[env_idx]
            seg_flat = seg_img.reshape(-1).to(torch.long)
            seg_valid = seg_flat[idx_valid]
            cls_ids = torch.full_like(seg_valid, -1, dtype=torch.long, device=self.device)
            mask_cls = torch.zeros_like(seg_valid, dtype=torch.bool, device=self.device)
            for seg_id, cidx in self.seg_id_to_cls.items():
                m = (seg_valid == seg_id)
                cls_ids[m] = cidx
                mask_cls |= m
            if mask_cls.sum() == 0:
                continue
            pts_xy = coords[mask_cls][:, :2]
            pts_idx_3D, infos = point_cloud_to_occ_idx_cnt(
                [pts_xy],
                env_max_cord[env_idx:env_idx+1],
                env_min_cord[env_idx:env_idx+1],
                self.sem_voxel_size[env_idx:env_idx+1],
                self.sem_map_size)
            if len(pts_idx_3D[0]) == 0:
                continue
            bound_mask = infos[0][0]
            reduced_ind = infos[0][1]
            cls_sel = cls_ids[mask_cls][bound_mask]
            num_vox = pts_idx_3D[0].shape[0]
            c = len(self.texts)
            current_votes = torch.zeros(num_vox, c, dtype=torch.float32, device=self.device)
            onehot = torch.nn.functional.one_hot(cls_sel, num_classes=c).to(torch.float32)
            torch_scatter.scatter_add(src=onehot, index=reduced_ind, out=current_votes, dim=0)
            xs = pts_idx_3D[0][:, 0]
            ys = pts_idx_3D[0][:, 1]
            self.votes[env_idx, :, xs, ys] += current_votes.T

    def _make_observation_cf(self, env_idx):
        sim = self.sem_occ_map[env_idx]
        sim = torch.softmax(sim, dim=-1)
        sim_cf = sim.permute(2, 0, 1).contiguous()
        return sim_cf

    def _make_groundtruth_cf(self, env_idx):
        v = self.votes[env_idx]
        total = v.sum(dim=0)
        maxvals, argmax = v.max(dim=0)
        ties = (v == maxvals.unsqueeze(0)).sum(dim=0) > 1
        k = self.dataset_cfg.vote_min_count_k
        uniform = torch.ones_like(v, dtype=torch.float32) / len(self.texts)
        out = torch.zeros_like(v, dtype=torch.float32)
        mask_uniform = (total < k) | ties
        out[:, mask_uniform] = uniform[:, mask_uniform]
        mask_onehot = ~mask_uniform
        idxs = torch.nonzero(mask_onehot, as_tuple=False)
        if idxs.shape[0] > 0:
            x = idxs[:, 0]
            y = idxs[:, 1]
            cls = argmax[mask_onehot]
            out[cls, x, y] = 1.0
        return out

    def _save_pair(self, env_idx, step_idx, observation_cf, groundtruth_cf):
        pose = self.pose[env_idx]
        classes = [t[0] for t in self.texts]
        item = {
            'env_idx': int(env_idx),
            'step': int(step_idx),
            'path': f'pair_env{env_idx}_step{step_idx}.pt'
        }
        data = {
            'observation': observation_cf.detach().to('cpu'),
            'groundtruth': groundtruth_cf.detach().to('cpu'),
            'pose': pose.detach().to('cpu'),
            'meta': {
                'classes': classes,
                'channels_first': True,
                'include_background': False,
                'alpha': float(self.dataset_cfg.alpha),
                'vote_min_count_k': int(self.dataset_cfg.vote_min_count_k)
            }
        }
        torch.save(data, os.path.join(self.run_dir, item['path']))
        self.dataset_index.append(item)

    def run(self, max_steps=None):
        steps = max_steps if max_steps is not None else self.dataset_cfg.max_steps
        self.stats['steps'] = int(steps)
        for step_idx in range(steps):
            self._nearby_sample_pose(step_idx)
            self._update_world_fusion()
            self._accumulate_gt_votes()
            for env_idx in range(self.num_envs):
                obs_cf = self._make_observation_cf(env_idx)
                gt_cf = self._make_groundtruth_cf(env_idx)
                self._save_pair(env_idx, step_idx, obs_cf, gt_cf)
                maxv = gt_cf.max(dim=0).values
                minv = gt_cf.min(dim=0).values
                c = len(self.texts)
                uniform_val = torch.tensor(1.0 / c, device=self.device)
                uniform_mask = torch.isclose(maxv, minv) & torch.isclose(maxv, uniform_val)
                onehot_mask = (maxv > 0.999) & (minv < 1e-6)
                self.stats['uniform_count'] += int(uniform_mask.sum().item())
                self.stats['onehot_count'] += int(onehot_mask.sum().item())
        index_path = os.path.join(self.run_dir, 'dataset_index.json')
        import json
        with open(index_path, 'w') as f:
            json.dump(self.dataset_index, f, indent=2)
        stats_path = os.path.join(self.run_dir, 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
