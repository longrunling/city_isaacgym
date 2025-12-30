from Envs.RLDrone.OccMapDrone import OccMapDrone
from Configs.RLDrone.Custom.Config import SemanticOccDrone_Config
from omegaconf import OmegaConf
import os
import matplotlib.pyplot as plt
import torch_scatter
import open3d as o3d
from matplotlib import cm

import sys
sys.path.insert(0, "Talk2DINO")
sys.path.insert(0, "Talk2DINO/src/open_vocabulary_segmentation")
from models.dinotext import DINOText
from models import build_model

import torch
import numpy as np
from Envs.utils import *

class SemanticOccDrone(OccMapDrone):

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

        self.config = SemanticOccDrone_Config

        self._init_semantic_encoder()
        self._get_text_feature()
        self._create_buffers_Semantic()
        self._calculate_gt_semantic_occ_map()
        
        # for debug
        self.visualize_ground_truth_sem_map(0)
        self.visualize_ground_truth_height_map(0)
        # self.render_forever()

    def _create_buffers_Semantic(self):
        
        # 语义占据图
        sem_map_size = self.config.sem_map_size
        self.sem_map_size = torch.tensor(
            sem_map_size, dtype=torch.int32, device=self.device)

        self.sem_occ_map = torch.zeros(
            self.num_envs, 
            sem_map_size[0], sem_map_size[1],
            len(self.texts),
            dtype=torch.float32, device=self.device, requires_grad=False)
        
        # 高度图
        self.height_map = torch.zeros(
            self.num_envs, 
            sem_map_size[0], sem_map_size[1],
            dtype=torch.float32, device=self.device, requires_grad=False)
        
        # 语义占据图的命中计数
        self.sem_occ_map_hit_cnt = torch.zeros(
            self.num_envs, 
            sem_map_size[0], sem_map_size[1],
            dtype=torch.float32, device=self.device, requires_grad=False)
        self.sem_voxel_size = (self.env_max_cord - self.env_min_cord)[:, :2] / self.sem_map_size

        # 前景占用
        self.fg_occ = torch.zeros_like(self.sem_occ_map_hit_cnt, requires_grad=False)

    def reset_idx(self, env_idx):
        '''
        重置env_idx指定的环境
        '''
        num_envs = len(env_idx)
        if num_envs == 0:
            return

        # 必须调用父类的reset_envs
        super().reset_idx(env_idx)

        # 重置语义占据图
        self.sem_occ_map[env_idx] = 0

        # 重置语义占据图命中计数
        self.sem_occ_map_hit_cnt[env_idx] = 0

        # 重置高度图
        self.height_map[env_idx] = 0

        # 重置前景索引
        self.fg_occ[env_idx] = 0

    def _init_semantic_encoder(self):

        device = self.rl_device
        config_path = 'Talk2DINO/src/open_vocabulary_segmentation/configs/cityscapes'
        config_file = 'dinotext_cityscapes_vitb_mlp_infonce.yml'
        path = os.path.join(config_path, config_file)
        cfg = OmegaConf.load(path)
        model = build_model(cfg.model)
        model.to(device).eval()

        self.palette = [
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [128, 128, 128]
        ]
        self.with_background = self.config.image_encoder.with_background
        if self.with_background:
            self.palette.insert(0, [0, 0, 0])
            model.with_bg_clean = True
        self.img_encoder = model

    def _get_text_feature(self):

        self.texts = self.config.image_encoder.classes
        self._calculate_class_idx()
        if len(self.texts) > len(self.palette):
            for _ in range(len(self.texts) - len(self.palette)):
                self.palette.append([np.random.randint(0, 255) for _ in range(3)])

        self.text_embeds = []
        with torch.no_grad():
            for class_group in self.texts:

                # text_embed = self.img_encoder.build_dataset_class_tokens(
                #     "sub_imagenet_template", class_group).to(self.device)
                # text_embed = self.img_encoder.build_text_embedding(text_embed)

                text_embed = self.img_encoder.encode_text_similarity(class_group)
                # [num_class, 1024]

                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                self.text_embeds.append(text_embed)

            group_index = []
            for i in range(len(self.texts)):
                len_class_group = self.text_embeds[i].shape[0]
                group_index.extend([i] * len_class_group)
            self.group_index = torch.tensor(group_index, device=self.device)

    def _calculate_class_idx(self):
        '''
        适配环境的class顺序
        '''
        env_class_id = self.simple_citygen._class_seg_id
        self.simple_city_gen_classes = self.config.image_encoder.simple_city_gen_classes
        new_texts = []
        for class_name in env_class_id.keys():
            if class_name in self.simple_city_gen_classes:
                idx = self.simple_city_gen_classes.index(class_name)
                new_texts.append(self.texts[idx])
        self.texts = new_texts    

    def _get_rgb_feature(self):

        with torch.no_grad():
            # 获取当前rgb图像的编码
            rgb_imgs = self.rgb_processed.clone()

            image_embed = self.img_encoder.encode_image_similarity(rgb_imgs)
            # [batch_size, 1024, 1024]
            image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
            batch_size = image_embed.shape[0]
            patches_x_row = image_embed.shape[1]**0.5
            image_embed = image_embed.reshape(
                batch_size, int(patches_x_row), int(patches_x_row), -1)
            self.image_embed = image_embed

    def _query_semantic(self):

        with torch.no_grad():
            batch_size = self.image_embed.shape[0]
            feature_size = self.image_embed.shape[-1]
            h, w = self.image_embed.shape[1], self.image_embed.shape[2]
            image_embed = self.image_embed.reshape(batch_size, -1, feature_size)

            temp_text_embed = torch.cat(self.text_embeds, dim=0)
            similarity = (image_embed @ temp_text_embed.T).squeeze(0)
            # [num_total_classes]

            index = self.group_index.expand(batch_size, h * w, -1).squeeze(0)
            dest = torch.ones(
                (batch_size, h * w, len(self.texts)),
                 dtype=similarity.dtype, device=self.device) * float('-inf')
            dest = dest.squeeze(0)
            similarity = torch.scatter_reduce(
                dest, dim=-1, index=index, src=similarity, reduce='max')
            similarity = similarity.reshape(batch_size, h, w, -1)

            # resize to the original image size
            h, w = self.rgb_processed.shape[2], self.rgb_processed.shape[3]
            self.interp_mode = 'bilinear'
            similarity = similarity.permute(0, 3, 1, 2)
            similarity = torch.nn.functional.interpolate(
                similarity,
                size =(h, w),
                mode=self.interp_mode,
                antialias=self.interp_mode in ["bilinear", "bicubic"])
            similarity = similarity.permute(0, 2, 3, 1)
            # similarity = torch.softmax(100 * similarity, dim = -1)
            self.similarity = similarity

    def _calculate_gt_semantic_occ_map(self):

        self.simple_citygen.sample_assets_point_cloud()
        self.gt_sem_occ_map, self.gt_height_map = \
            self.simple_citygen.create_ground_truth_sem_map()

    def update_semantic_occ_map(self):
        '''
        更新语义占据图与高度图
        需要先调用update_visual()获取最新的rgb图像
        '''
        # 获取当前rgb图像的编码
        self._get_rgb_feature()

        # 查询语义相似度,存储在self.similarity
        self._query_semantic()

        # 全部点云
        self.get_point_cloud(min_seg_id=0)

        # 包含地面的掩码
        seg_mask = self.get_seg_id_mask(min_seg_id=1)
        # 过滤depth值为0的位置
        new_seg_mask = []
        for idx in range(self.num_envs):
            new_seg_mask.append(seg_mask[idx][self.fg_coord[idx, :]])
        seg_mask = new_seg_mask

        # 语义掩码
        seg_mask_fg = self.get_seg_id_mask(min_seg_id=2)
        new_seg_mask_fg = []
        for idx in range(self.num_envs):
            new_seg_mask_fg.append(seg_mask_fg[idx][self.fg_coord[idx, :]])
        seg_mask_fg = new_seg_mask_fg

        # 全部点云
        pts_target = [
            pt[seg_mask[idx]] if pt.shape[0] > 0 else pt
            for idx, pt in enumerate(self.coords_world)
        ]

        # 前景点云
        pts_target_fg = [
            pt[seg_mask_fg[idx]] if pt.shape[0] > 0 else pt
            for idx, pt in enumerate(self.coords_world)
        ]

        # 获取高度信息
        heights = [
            pts_target[idx][:, 2] if pts_target[idx].shape[0] > 0 
            else torch.tensor([], device=self.device, dtype=torch.float32)
            for idx in range(self.num_envs)
        ]

        # 只保留xy坐标
        pts_target = [
            pts_target[idx][:, :2] if pts_target[idx].shape[0] > 0 
            else torch.tensor(
                [], device=self.device, dtype=torch.float32).reshape(-1, 2) 
            for idx in range(self.num_envs)
        ]
        pts_target_fg = [
            pts_target_fg[idx][:, :2] if pts_target_fg[idx].shape[0] > 0 
            else torch.tensor(
                [], device=self.device, dtype=torch.float32).reshape(-1, 2) 
            for idx in range(self.num_envs)
        ]
        
        # 将点云投影到体素网格中，返回体素坐标以及每个点云对应的体素索引
        env_max_cord = self.env_max_cord[:, :2]
        env_min_cord = self.env_min_cord[:, :2]
        pts_idx_all, infos = point_cloud_to_occ_idx_cnt(
            pts_target,
            env_max_cord,
            env_min_cord,
            self.sem_voxel_size,
            self.sem_map_size)
        
        pts_idx_all_fg, infos_fg = point_cloud_to_occ_idx_cnt(
            pts_target_fg,
            env_max_cord,
            env_min_cord,
            self.sem_voxel_size,
            self.sem_map_size)        

        # 获取前景相似度
        fg_similarity = self.similarity.reshape(
            self.num_envs, -1, len(self.texts))
        fg_similarity = [
            fg_similarity[idx, self.fg_coord[idx], :] 
            for idx in range(self.num_envs)
        ]
        fg_similarity = [
            fg_similarity[idx][seg_mask_fg[idx], :] 
            for idx in range(self.num_envs)
        ]

        # 将前景相似度聚合到体素网格中
        for env_idx in range(self.num_envs):
            
            # 使用全部点云更新高度
            pts_idx_3D = pts_idx_all[env_idx]   
            # [num_point, 3]

            if (isinstance(pts_idx_3D, list) and \
                len(pts_idx_3D) == 0) or \
                pts_idx_3D.shape[0] == 0:
                continue

            # 当前高度
            heights_env = heights[env_idx]

            # 使用bound_masks过滤超出范围的相似度
            bound_mask = infos[env_idx][0]
            heights_env = heights_env[bound_mask]

            # 聚合同位置的高度
            accumulate_idx = infos[env_idx][1]
            # 初始化current_heights为之前对应位置的高度
            current_heights = self.height_map[env_idx,
                pts_idx_3D[:, 0], pts_idx_3D[:, 1]].clone()
            torch_scatter.scatter(
                src=heights_env, index=accumulate_idx, out=current_heights, reduce='max', dim=0)

            # height最小值,以区分探索和未探索
            current_heights = torch.clamp(current_heights, min=self.config.min_height)

            # 更新height
            self.height_map[env_idx,
                pts_idx_3D[:, 0], pts_idx_3D[:, 1]] = current_heights
            
            # 使用前景点云更新语义相似度
            pts_idx_3D_fg = pts_idx_all_fg[env_idx]

            if (isinstance(pts_idx_3D_fg, list) and \
                len(pts_idx_3D_fg) == 0) or \
                pts_idx_3D_fg.shape[0] == 0:
                continue            
            
            # 通过accumulate_idx将点云相似度聚合到当前体素网格中
            fg_similarity_env = fg_similarity[env_idx]
            # 使用bound_masks过滤超出范围的相似度
            bound_mask = infos_fg[env_idx][0]
            fg_similarity_env = fg_similarity_env[bound_mask]

            accumulate_idx = infos_fg[env_idx][1]  
            current_similarity = torch.zeros(
                len(pts_idx_3D_fg), len(self.texts),
                dtype=torch.float32, device=self.device, requires_grad=False)
            torch_scatter.scatter(
                src=fg_similarity_env, index=accumulate_idx, out=current_similarity, reduce='mean', dim=0)

            # 聚合到全局语义占据图中，根据命中数加权
            # 计算之前相似度
            previous_similarity = self.sem_occ_map[
                env_idx, pts_idx_3D_fg[:, 0], pts_idx_3D_fg[:, 1], :3]
            previous_hit_cnt = self.sem_occ_map_hit_cnt[
                env_idx, pts_idx_3D_fg[:, 0], pts_idx_3D_fg[:, 1]]

            # 当前命中数
            current_hit_cnt = infos_fg[env_idx][2]

            # 加权（平滑更新策略,alpha为前后权值）
            alpha = self.config.alpha
            current_hit_cnt_weighted = alpha * current_hit_cnt
            previous_hit_cnt_weighted = (1 - alpha) * previous_hit_cnt
            weight_sum = current_hit_cnt_weighted + previous_hit_cnt_weighted

            current_similarity_weighted = \
                current_similarity * current_hit_cnt_weighted.reshape(-1, 1)
            previous_similarity_weighted = \
                previous_similarity * previous_hit_cnt_weighted.reshape(-1, 1)

            update_similarity = \
                current_similarity_weighted + previous_similarity_weighted
            update_similarity = update_similarity / weight_sum.reshape(-1, 1)

            # 更新相似度
            self.sem_occ_map[
                env_idx,
                pts_idx_3D_fg[:, 0], pts_idx_3D_fg[:, 1], :3] = update_similarity
            # 更新命中数
            self.sem_occ_map_hit_cnt[
                env_idx, 
                pts_idx_3D_fg[:, 0], pts_idx_3D_fg[:, 1]] = weight_sum
            # 更新前景占用
            self.fg_occ[env_idx, pts_idx_3D_fg[:, 0], pts_idx_3D_fg[:, 1]] = 1

    def visualize_ground_truth_sem_map(self, env_idx):

        # 可视化语义地图
        semantic_map = self.gt_sem_occ_map[env_idx]
        vis_semantic_map = torch.zeros(
            semantic_map.shape[0], semantic_map.shape[1],
            dtype=torch.int32, device=self.device)
        for i in range(0, self.class_num):
            mask = (vis_semantic_map == 0)
            vis_semantic_map += semantic_map[:, :, i] * (i + 1) * mask
        vis_semantic_map = vis_semantic_map.cpu().numpy()

        plt.figure()
        plt.imshow(vis_semantic_map, cmap='tab20')
        plt.title('Ground Truth Semantic Map')
        plt.axis('off')
        plt.colorbar()
        plt.savefig('ground_truth_semantic_map.png', dpi=300)
        plt.close()

    def visualize_ground_truth_height_map(self, env_idx):

        height_map = self.gt_height_map[env_idx, :, :].cpu().numpy()

        plt.figure(figsize=(10, 10))
        # plt.axis('off')
        plt.imshow(height_map, cmap='hot')
        plt.colorbar()
        plt.tight_layout()
        plt.title('Height Map')
        plt.savefig('ground_truth_height_map.png', dpi=300)
        plt.close()     

    def visualize_similarity(self, env_idx, class_idx):

        similarity_vis = self.similarity.cpu().numpy()[env_idx, :, :, class_idx]
        # patches_x_row = similarity_vis.shape[0]**0.5
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        # plt.imshow(
        #     similarity_vis.reshape(int(patches_x_row), int(patches_x_row)), cmap='magma')
        plt.imshow(similarity_vis, cmap='magma')
        plt.colorbar()
        plt.tight_layout()
        plt.title('Similarity on class: {}'.format(self.texts[class_idx][0]))
        plt.savefig('similarity_vis.png')
        plt.close()

    def visualize_semantic_occ_map_soft_max(self, env_idx, class_name):

        class_idx = self.texts.index(class_name)
        similarity_values = self.sem_occ_map[env_idx, :, :, :]
        similarity_values = torch.softmax(100 * similarity_values, dim=-1)
        similarity_values = torch.argmax(similarity_values, dim=-1)

        # 只显示指定类别
        sem_map_vis = torch.zeros_like(similarity_values)
        sem_map_vis[similarity_values == class_idx] = 1
        sem_map_vis = sem_map_vis * self.fg_occ[env_idx]
        sem_map_vis = sem_map_vis.cpu().numpy()

        plt.figure(figsize=(10, 10))
        # plt.axis('off')
        plt.imshow(sem_map_vis, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        plt.title('Explored Sem_map on class: {}'.format(self.texts[class_idx][0]))
        plt.savefig('semantic_occ_map_vis.png')
        plt.close()

    def visualize_semantic_occ_map_threshold(self, env_idx, class_name, threshold=0.5):

        class_idx = self.texts.index(class_name)
        similarity_values = self.sem_occ_map[env_idx, :, :, class_idx]
        similarity_values = similarity_values * self.fg_occ[env_idx]

        # 进行min-max归一化
        min_val = torch.min(similarity_values)
        max_val = torch.max(similarity_values)
        similarity_values = (similarity_values - min_val) / (max_val - min_val + 1e-8)

        # 小于阈值的设为0
        sem_map_vis = torch.zeros_like(similarity_values)
        sem_map_vis[similarity_values >= threshold] = 1
        sem_map_vis = sem_map_vis.cpu().numpy()

        plt.figure(figsize=(10, 10))
        # plt.axis('off')
        plt.imshow(sem_map_vis, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        plt.title('Explored Sem_map on class: {}'.format(self.texts[class_idx][0]))
        plt.savefig('semantic_occ_map_vis.png')
        plt.close()

    def visualize_height_map(self, env_idx):

        height_map = self.height_map[env_idx, :, :].cpu().numpy()

        plt.figure(figsize=(10, 10))
        # plt.axis('off')
        plt.imshow(height_map, cmap='hot')
        plt.colorbar()
        plt.tight_layout()
        plt.title('Height Map')
        plt.savefig('height_map_vis.png')
        plt.close()






