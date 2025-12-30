from rl_games.algos_torch import network_builder
import torch.nn as nn
import torch


class RLDroneBuilder(network_builder.A2CBuilder):        
    '''
    Add a custom feature extractor for the drone's observations.
    '''
    def __init__(self):
        super().__init__()

    class Network(network_builder.A2CBuilder.Network):

        def __init__(self, params, **kwargs):
            
            # 初始化actor和critic网络
            feature_size = params['feature_size']
            super_kwargs = kwargs.copy()
            super_kwargs['input_shape'] = (feature_size,)
            super().__init__(params, **super_kwargs)

            input_shape = kwargs.pop('input_shape')
            occ_map_shape: tuple = input_shape['occ_map']
            state_shape: tuple = input_shape['state']
            self.AT_NUM = kwargs['actions_num']
            action_size: int = len(self.AT_NUM)

            # 重写rl_games的action头
            if len(self.units) == 0:
                out_size = feature_size
            else:
                out_size = self.units[-1]
            action_logits = nn.Linear(out_size, sum(self.AT_NUM))
            self.logits = torch.nn.ModuleList([action_logits])

            # 自定义特征提取器
            self.naive_encoder_occ = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
            )

            dummy_input = torch.zeros(occ_map_shape)
            dummy_input.unsqueeze_(0).unsqueeze_(1)  
            # 增加 batch 维度和 channel 维度

            with torch.no_grad():
                conv_out = self.naive_encoder_occ(dummy_input)
                output_layer_size = conv_out.reshape(dummy_input.shape[0], -1)
                output_layer_size = output_layer_size.shape[1]

            self.output_layer_occ = nn.Sequential(
                nn.Linear(in_features = output_layer_size, 
                          out_features = feature_size, bias=True),   
                # feature size after 3D CNN, grid_size = 20
                nn.ReLU(inplace=True),
            )

            dummy_input = torch.zeros(state_shape).view(1, -1, action_size)
            pos_encoding = self.positional_encoding(dummy_input, freqs=2).view(1, -1)
            in_feature_size = pos_encoding.shape[1]                          

            self.naive_encoder_state = nn.Sequential(
                nn.Linear(in_features = in_feature_size, 
                          out_features = feature_size, bias=True),   
                # action_size (positional embedding) * buffer_size
                nn.ReLU(inplace=True),
                nn.Linear(in_features = feature_size, 
                          out_features = feature_size, bias=True),
                nn.ReLU(inplace=True),
            )

            self.output_layer = nn.Sequential(
                nn.Linear(in_features = 2 * feature_size, 
                          out_features = feature_size, bias=True),
                nn.ReLU(inplace=True),
            )

        def positional_encoding(self, positions, freqs=2):
            """
            Params:
                positions: [num_env, buffer_size, action_size]

            Return:
                pts: [num_env, buffer_size, 4 * action_size]
            """
            freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  
            # [2]
            pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1], ))  
            # [num_env, buffer_size, 2*action_size]
            pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
            return pts

        def forward(self, obs_dict):

            obs = obs_dict['obs']
            state = obs['state']
            occ_map = obs['occ_map']
            num_env = state.shape[0]

            # state
            state = self.positional_encoding(state).view(num_env, -1)
            feature_state = self.naive_encoder_state(state)    
            # [num_env, feature_size]

            # occ_map
            occ_map = occ_map.unsqueeze(1)  
            # [num_env, 1, X, Y, Z]
            feature_occ = self.naive_encoder_occ(occ_map).reshape(num_env, -1) 
            feature_occ = self.output_layer_occ(feature_occ)
            # [num_env, feature_size]

            # hybrid feature
            feature_hybrid = torch.cat((feature_state, feature_occ), dim=-1)
            feature_hybrid = self.output_layer(feature_hybrid)
            # [num_env, feature_size]

            new_obs_dict = obs_dict.copy()
            new_obs_dict['obs'] = feature_hybrid
            logits, value, states = super().forward(new_obs_dict)

            # 将logits还原为动作
            logits = logits[0]
            logits = [split for split in torch.split(logits, tuple(self.AT_NUM), dim=1)]

            return logits, value, states

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return RLDroneBuilder.Network(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)        