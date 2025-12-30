import torch
import torch.nn as nn

from skrl.models.torch import Model, MultiCategoricalMixin, DeterministicMixin

# define the shared model
class SharedModel(MultiCategoricalMixin, DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device,
                 input_shape, feature_size = 256,
                 unnormalized_log_prob: bool = True, reduction: str = "sum",
                 clip_actions: bool = False):
        
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob, reduction, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        self.input_shape = input_shape

        # shared feature extraction
        self.naive_encoder_occ = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        occ_map_shape = self.input_shape['occ_map']
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

        state_shape = self.input_shape['state']
        action_size = self.input_shape['state'][1]
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

        # separated layers ("policy")
        self.mean_layer = nn.Linear(feature_size, self.num_actions)

        # separated layer ("value")
        self.value_layer = nn.Linear(feature_size, 1)

    def positional_encoding(self, positions, freqs=2):
        """
        Params:
            positions: [num_env, buffer_size, action_size]

        Return:
            pts: [num_env, buffer_size, 4 * action_size]
        """
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # [2]
        pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1], ))  # [num_env, buffer_size, 2*action_size]
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts    

    # override the .act(...) method to disambiguate its call
    def act(self, inputs, role):
        if role == "policy":
            return MultiCategoricalMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role):
        if role == "policy":
            # save shared layers/network output to perform a single forward-pass
            self._shared_output = self.compute_feature(inputs)
            return self.mean_layer(self._shared_output), {}
        elif role == "value":
            # use saved shared layers/network output to perform a single forward-pass, if it was saved
            shared_output = self.compute_feature(inputs) \
                if self._shared_output is None else self._shared_output
            self._shared_output = None  # reset saved shared output to prevent the use of erroneous data in subsequent steps
            return self.value_layer(shared_output), {}
        
    def compute_feature(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [num_env, buffer_size*action_size + 1*X*Y*Z + k*H*W]
        "value = value.permute(0, 4, 1, 2, 3).reshape(value.shape[0], -1)" in wrapper
        """
        input = inputs['states']
        num_env = input.shape[0]

        # action
        state_size = self.input_shape['state'][0] * self.input_shape['state'][1]
        action_input = input[:, :state_size]
        # [num_env, buffer_size*action_size]
        action_input = action_input.view(num_env, -1, self.input_shape['state'][1])
        action_input = self.positional_encoding(action_input).view(num_env, -1)
        feature_action = self.naive_encoder_state(action_input)    # [num_env, 256]

        # grid
        grid_input = input[:, state_size:state_size + 8000]     
        # [num_env, 1*X*Y*Z]
        reshape_size = [num_env, 1] + list(self.input_shape['occ_map'])
        # [num_env, 1, X, Y, Z]
        grid_input = grid_input.reshape(*reshape_size)

        feature_grid = self.naive_encoder_occ(grid_input).reshape(num_env, -1) 
        # [num_env, hidden_layer_size]
        feature_grid = self.output_layer_occ(feature_grid)  
        # [num_env, 256]

        feature_hybrid = self.output_layer(
            torch.cat((feature_action, feature_grid), dim=-1))  
        # [num_env, 256*3] -> [num_env, 256]

        return feature_hybrid