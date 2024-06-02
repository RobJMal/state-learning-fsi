import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder parameters 
depth = 32
stride = 2 
kernel_size = 4
activation = "ReLU"
observation_shape = (3, 128, 128)

class Pixel2StateNet(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()

        activation = getattr(nn, activation)()
        self.observation_shape = observation_shape

        self.encoder_network = nn.Sequential(
            nn.Conv2d(
                self.observation_shape[0],
                depth * 1,
                kernel_size,
                stride,
            ),
            activation,
            nn.Conv2d(
                depth * 1,
                depth * 2,
                kernel_size,
                stride,
            ),
            activation,
            nn.Conv2d(
                depth * 2,
                depth * 4,
                kernel_size,
                stride,
            ),
            activation,
            nn.Conv2d(
                depth * 4,
                depth * 8,
                kernel_size,
                stride,
            ),
            activation,
        )

        feature_map_size = 0
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape)
            dummy_output = self.encoder_network(dummy_input)
            feature_map_size = dummy_output.view(1, -1).size(1)
            print(f"Feature map size: ", feature_map_size)

        num_proprio_states = 24 # Fish-swim environment has 24 observable states

        # Network for mapping between encoded pixel input to state
        self.mlp_network = nn.Sequential(
            nn.Linear(feature_map_size, 4092),
            nn.BatchNorm1d(4092),
            nn.Dropout(0.1),
            activation,
            nn.Linear(4092, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            activation,
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            activation,
            nn.Linear(512, num_proprio_states),
        )


    def forward(self, x):
        x = self.encoder_network_forward(self.network, x, input_shape=self.observation_shape)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp_network(x)

        return x


    def encoder_network_forward(self, network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
        batch_with_horizon_shape = x.shape[: -len(input_shape)]
        if not batch_with_horizon_shape:
            batch_with_horizon_shape = (1,)
        if y is not None:
            x = torch.cat((x, y), -1)
            input_shape = (x.shape[-1],)
        x = x.reshape(-1, *input_shape)
        x = network(x)

        x = x.reshape(*batch_with_horizon_shape, *output_shape)
        return x
