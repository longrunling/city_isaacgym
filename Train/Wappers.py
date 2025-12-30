from Configs.RLDrone.Custom.Config import Train_Config
from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver, MultiObserver
from rl_games.torch_runner import Runner
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from rl_games.algos_torch import model_builder


def pre_process_train_config(cfg):
    pass

def register_network(network_name, network_class):
    model_builder.register_network(network_name, network_class)

def build_runner(cfg):

    config_dict = omegaconf_to_dict(cfg.train)
    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device
    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None
    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    observers = [RLGPUAlgoObserver()]
    runner = Runner(MultiObserver(observers))
    runner.load(config_dict)
    runner.reset()
    return runner