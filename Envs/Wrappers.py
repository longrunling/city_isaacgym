import os
from Configs.RLDrone.Custom.Config import Env_Config
import yaml
from rl_games.common import env_configurations, vecenv
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv
from isaacgymenvs.utils.utils import set_seed

def pre_process_env_config(cfg):

    for key, value in vars(Env_Config).items():
        if key in cfg:
            cfg[key] = value

    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

def get_rlgames_env_creator(VecTask_cfg, env_class):

    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """
        if Env_Config.multi_gpu:

            # local rank of the GPU in a node
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            world_size = int(os.getenv("WORLD_SIZE", "1"))

            print(f"global_rank = {global_rank} local_rank = {local_rank} world_size = {world_size}")

            sim_device = f'cuda:{local_rank}'
            rl_device = f'cuda:{local_rank}'

        else:
            sim_device = Env_Config.sim_device
            rl_device = Env_Config.rl_device

        # create native task and pass custom config
        env = env_class(VecTask_cfg, rl_device, sim_device,
                Env_Config.graphics_device_id, Env_Config.headless)
        return env
    return create_rlgpu_env

def register_env(cfg, env_class):

    VecTask_cfg = cfg.task
    
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': get_rlgames_env_creator(VecTask_cfg, env_class),
    })

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))