from Envs.RLDrone.RLDrone import RLDrone
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
from sklearn.decomposition import PCA
import yaml

import skrl

def test_point_cloud():

    mesh_folder = 'Datas_city/building/obj/2.obj'
    mesh = o3d.io.read_triangle_mesh(mesh_folder)
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    # o3d.visualization.draw_geometries([pcd])
    # print('ok')

    visualize_pca_components(pcd)

def visualize_pca_components(pcd):
    points = np.asarray(pcd.points)
    centered_points = points - np.mean(points, axis=0)
    
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    components = pca.components_
    
    arrow_scale = 1 * np.max(np.linalg.norm(points, axis=1))
    origin = np.mean(points, axis=0)
    
    arrows = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    for i in range(3):
        # 构造旋转矩阵（主成分方向 + 两个正交方向）
        z_axis = components[i]  # 主成分方向
        x_axis = np.cross([1, 0, 0], z_axis)  # 尝试与 [1,0,0] 叉积
        if np.linalg.norm(x_axis) < 1e-6:  # 如果平行，换另一个方向
            x_axis = np.cross([0, 1, 0], z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)  # 归一化
        y_axis = np.cross(z_axis, x_axis)  # 确保右手坐标系
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3 旋转矩阵
        
        # 创建箭头
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005 * arrow_scale,
            cone_radius=0.01 * arrow_scale,
            cylinder_height=0.8 * arrow_scale,
            cone_height=0.2 * arrow_scale
        )
        arrow.paint_uniform_color(colors[i])
        arrow.translate(origin)
        arrow.rotate(rotation_matrix, center=origin)  # 正确调用 rotate()
        arrows.append(arrow)
    
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd, *arrows], window_name="PCA主成分分析")

def test_SB3():

    from Configs.RLDrone.pre_process_cfg import pre_process_cfg_skrl

    VecTask_cfg = pre_process_cfg_skrl()

    from Configs.RLDrone.Custom.Config import Env_Config
    test = RLDrone(
        VecTask_cfg, Env_Config.rl_device, Env_Config.sim_device,
        Env_Config.graphics_device_id, Env_Config.headless)

    from Envs.Wrappers_SB3 import EnvWrapperSB3
    env_sb3 = EnvWrapperSB3(test)

    from stable_baselines3.common.policies import ActorCriticPolicy_Train_Eval
    from Train.Network_SB3 import Hybrid_Encoder
    # ===== Setup the config =====
    config = dict(
        algo=dict(
            policy=ActorCriticPolicy_Train_Eval,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=Hybrid_Encoder,
                features_extractor_kwargs=dict(
                    feature_dim = 256,
                    state_input_shape = (env_sb3.config.buffer_size * 6,)
                )
            ),
            env=env_sb3,
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=0.05,
            max_grad_norm=1,
            n_steps=128,  # steps to collect in each env
            n_epochs=5,
            batch_size=128,
            clip_range=0.2,
            vf_coef=0.8,
            clip_range_vf=0.2,
            ent_coef=0.01,
            tensorboard_log='./runs/',
            create_eval_env=False,
            verbose=2,
            seed=1,
            device='cuda:0',
        ),

        # Meta data
        gpu_simulation=True,
        # project_name=project_name,
        # team_name=team_name,
        # exp_name=exp_name,
        # seed=seed,
        # use_wandb=use_wandb,
        # trial_name=trial_name,
        # log_dir=log_dir
    )    

    from stable_baselines3.ppo.ppo_grid_obs import PPO_Grid_Obs
    model = PPO_Grid_Obs(**config["algo"])

    save_freq = 10000
    eval_freq = 5000

    # callbacks = [
    #     BestCKPTCallback(
    #         name_prefix="rl_model",
    #         verbose=1,
    #         save_freq=save_freq,
    #         save_path=os.path.join('./runs/', "models"),
    #         key_list=["episode_reward"]
    #     )
    # ]    

    model.learn(
        # training
        total_timesteps = env_sb3.num_envs * 128 * 1000,
        # args.num_envs * args.n_steps * args.total_iters,    # num_steps_per_iter: args.num_envs * args.n_steps
        # callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=None,
        eval_freq=eval_freq,
        n_eval_episodes=50,
        eval_log_path=None,

        # logging
        tb_log_name='test',  # Should place the algorithm name here!
        log_interval=1,
    )

def demo_test():

    from Configs.RLDrone.pre_process_cfg import pre_process_cfg_skrl

    VecTask_cfg = pre_process_cfg_skrl()

    from Configs.RLDrone.Custom.Config import Env_Config
    test = RLDrone(
        VecTask_cfg, Env_Config.rl_device, Env_Config.sim_device,
        Env_Config.graphics_device_id, Env_Config.headless)
    
    test.reset()

    test.get_observations()

    # test.visualize_point_cloud(3)
    test.visualize_rgb_image_cv2(3)
    # test.visualize_depth_image(3)
    # test.visualize_occ_map(3)

    occ_map = test.occ_map[3]
    # 对occ_map进行上下与左右翻转
    occ_map = torch.flip(occ_map, dims=[0, 1])    

    occ_map = torch.sum(occ_map, dim=2)
    occ_map = torch.clip(occ_map, 0, 1)

    occ_map_np = occ_map.cpu().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(occ_map_np, cmap='hot')
    # plt.colorbar()
    # plt.title('2D Occupancy Map Slice')
    # plt.axis('off')
    plt.xticks(ticks=[0,10,19], labels=['0','10','20'])
    plt.yticks(ticks=[0,10,19], labels=['0','10','20'])
    plt.xlabel('m')
    plt.ylabel('m')
    plt.savefig('test.png')


    # test.render_forever()

def test_skrl():

    from utils.logger import CompleteLogger
    logger = CompleteLogger("Log")    

    from Envs.Wrappers_skrl import EnvWrapperSKRL

    from Configs.RLDrone.pre_process_cfg import pre_process_cfg_skrl
    VecTask_cfg = pre_process_cfg_skrl()

    from Configs.RLDrone.Custom.Config import Env_Config
    test = RLDrone(
        VecTask_cfg, Env_Config.rl_device, Env_Config.sim_device,
        Env_Config.graphics_device_id, Env_Config.headless)
    env = EnvWrapperSKRL(test)
    from skrl.envs.wrappers.torch import wrap_env
    env = wrap_env(env, wrapper="isaacgym-preview4")

    from Train.Network_skrl import SharedModel

    models = {}
    models["policy"] = SharedModel(
        env.observation_space, env.action_space, 
        env.device, env.input_shape)
    models["value"] = models["policy"]

    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

    # adjust some configuration if necessary
    cfg_agent = PPO_DEFAULT_CONFIG.copy()
    cfg_agent["rollouts"] = 128
    cfg_agent["learning_epochs"] = 5
    cfg_agent["mini_batch_size"] = 128
    cfg_agent["learning_rate"] = 1e-4
    cfg_agent["grad_norm_clip"] = 1.0
    cfg_agent["clip_predicted_values"] = True
    cfg_agent["entropy_loss_scale"] = 0.01
    cfg_agent["value_loss_scale"] = 0.8
    cfg_agent["kl_threshold"] = 0.05

    cfg_agent["experiment"]["directory"] = "./runs/"
    # cfg_agent["experiment"]["experiment_name"] = "test"
    cfg_agent["experiment"]["write_interval"] = 1

    # instantiate the agent
    # (assuming a defined environment <env> and memory <memory>)
    from skrl.memories.torch import RandomMemory
    memory = RandomMemory(
        memory_size=cfg_agent["rollouts"], 
        num_envs=env.num_envs, device=env.device)

    agent = PPO(models=models,
                cfg=cfg_agent,
                memory=memory,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)
    
    from skrl.trainers.torch import SequentialTrainer

    # assuming there is an environment called 'env'
    # and an agent or a list of agents called 'agents'

    # create a sequential trainer
    cfg = {"timesteps": 1000 * cfg_agent["rollouts"], "headless": True}
    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)

    # train the agent(s)
    trainer.train()

def test_Sem_drone():

    from Configs.RLDrone.pre_process_cfg import pre_process_cfg_skrl
    VecTask_cfg = pre_process_cfg_skrl()    

    from Configs.RLDrone.Custom.Config import Env_Config
    from Envs.RLDrone.RLDrone_Sem import RLDrone_Sem
    from Envs.RLDrone.SemanticOccDrone import SemanticOccDrone

    test = RLDrone_Sem(
        VecTask_cfg, Env_Config.rl_device, Env_Config.sim_device,
        Env_Config.graphics_device_id, Env_Config.headless)

    from Envs.Wrappers_SB3 import EnvWrapperSB3
    env_sb3 = EnvWrapperSB3(test)

    from stable_baselines3.common.policies import ActorCriticPolicy_Train_Eval
    from Train.Network_SB3 import Hybrid_Encoder, Height_Map_Encoder
    # ===== Setup the config =====
    config = dict(
        algo=dict(
            policy=ActorCriticPolicy_Train_Eval,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=Height_Map_Encoder,
                features_extractor_kwargs=dict(
                    feature_dim = 256,
                    state_input_shape = {
                        'state': (test.buffer_size, test.action_size),
                        'occ_map': tuple(test.sem_map_size.tolist())
                    }
                )
            ),
            env=env_sb3,
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=0.2,
            max_grad_norm=1,
            n_steps=128,  # steps to collect in each env
            n_epochs=5,
            batch_size=128,
            clip_range=0.2,
            vf_coef=0.8,
            clip_range_vf=0.2,
            ent_coef=0.01,
            tensorboard_log='./runs/',
            create_eval_env=False,
            verbose=2,
            seed=1,
            device='cuda:0',
        ),

        # Meta data
        gpu_simulation=True,
    )    

    from stable_baselines3.ppo.ppo_grid_obs import PPO_Grid_Obs
    model = PPO_Grid_Obs(**config["algo"])

    save_freq = 10000
    eval_freq = 5000

    model.learn(
        # training
        total_timesteps = env_sb3.num_envs * 128 * 1000,
        # args.num_envs * args.n_steps * args.total_iters,    
        # num_steps_per_iter: args.num_envs * args.n_steps
        # callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=None,
        eval_freq=eval_freq,
        n_eval_episodes=50,
        eval_log_path=None,

        # logging
        tb_log_name='height_map',  # Should place the algorithm name here!
        log_interval=1,
    )

def test_sem_drone_eval():

    from Configs.RLDrone.pre_process_cfg import pre_process_cfg_skrl
    VecTask_cfg = pre_process_cfg_skrl()    

    from Configs.RLDrone.Custom.Config import Env_Config
    from Envs.RLDrone.RLDrone_Sem import RLDrone_Sem
    from Envs.RLDrone.RLDrone_Sem_eval import RLDrone_Sem_eval
    from Envs.RLDrone.BaseDroneEnv import simple_citygen

    env_gen = simple_citygen()
    paras = {'simple_citygen': env_gen}

    train_env = RLDrone_Sem(
        VecTask_cfg, Env_Config.rl_device, Env_Config.sim_device,
        Env_Config.graphics_device_id, Env_Config.headless, **paras)
    
    from Envs.Wrappers_SB3 import EnvWrapperSB3
    train_env_sb3 = EnvWrapperSB3(train_env)

    eval_env = RLDrone_Sem_eval(
        VecTask_cfg, Env_Config.rl_device, Env_Config.sim_device,
        Env_Config.graphics_device_id, Env_Config.headless, **paras)
    
def test_data_collection():
    
    from Envs.Wrappers import pre_process_env_config, register_env
    from Train.Wappers import pre_process_train_config, register_network, build_runner
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    import os 
    from Envs.Wrappers_SB3 import EnvWrapperSB3

    # set_np_formatting()

    from Configs.RLDrone.pre_process_cfg import pre_process_cfg_skrl
    VecTask_cfg = pre_process_cfg_skrl()

    from Configs.RLDrone.Custom.Config import Env_Config
    # test = RLDrone(
    #     VecTask_cfg, Env_Config.rl_device, Env_Config.sim_device,
    #     Env_Config.graphics_device_id, Env_Config.headless)
    from Envs.RLDrone.DataCollectionEnv import DataCollectionEnv
    test = DataCollectionEnv(
        VecTask_cfg, Env_Config.rl_device, Env_Config.sim_device,
        Env_Config.graphics_device_id, Env_Config.headless)

    # test.reset()
    test.collect_data()

 

    
if __name__ == "__main__":

    # test_base_drone()

    # demo_test()

    # test_SB3()

    # test_skrl()

    # test_sem_drone_eval()

    test_data_collection()