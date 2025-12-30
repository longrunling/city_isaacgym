from Configs.RLDrone.Custom.Config import Env_Config, VecTask_cfg
import skrl
from isaacgymenvs.utils.utils import set_seed

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

def pre_process_cfg_skrl():

    # local rank of the GPU in a node
    local_rank = skrl.config.torch.local_rank
    # global rank of the GPU
    global_rank = skrl.config.torch.rank
    # total number of GPUs across all nodes
    world_size = skrl.config.torch.world_size    

    if Env_Config.multi_gpu:

        print(f"global_rank = {global_rank} local_rank = {local_rank} world_size = {world_size}")

        Env_Config.sim_device = f'cuda:{local_rank}'
        Env_Config.rl_device = f'cuda:{local_rank}'
        Env_Config.graphics_device_id = local_rank

    Env_Config.seed = set_seed(Env_Config.seed, rank=local_rank)

    cfg = class_to_dict(VecTask_cfg)
    return cfg