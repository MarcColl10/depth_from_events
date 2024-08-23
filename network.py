import torch
import torch.nn as nn

from blocks import conv_encoder, global_pool_decoder, LazyConvGru, upsample_decoder
from network_utils import NetworkWrapper


class GlobalFlowNetwork(nn.Module):

    def __init__(self, encoder_channels, memory_channels, scaling):
        super().__init__()

        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels)
        self.memory = LazyConvGru(memory_channels, 3)
        self.decoder = global_pool_decoder(2)

    def forward(self, input, hidden=None):
        input = input[:, :2]  # only polarity channels
        encoder = self.encoder(input)
        memory = self.memory(encoder, hidden)
        flow = self.decoder(memory)

        flow *= self.scaling
        flow_map = torch.ones_like(input) * flow.unsqueeze(-1).unsqueeze(-1)

        return flow_map, memory


class WrappedGlobalFlowNetwork(NetworkWrapper, GlobalFlowNetwork):
    pass


class FlowNetwork(nn.Module):

    def __init__(self, encoder_channels, memory_channels, decoder_channels, scaling):
        super().__init__()

        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels)
        self.memory = LazyConvGru(memory_channels, 3)
        self.decoder = upsample_decoder(decoder_channels)

    def forward(self, input, hidden=None):
        input = input[:, :2]  # only polarity channels
        encoder = self.encoder(input)
        memory = self.memory(encoder, hidden)
        flow_map = self.decoder(memory)

        flow_map *= self.scaling

        return flow_map, memory


class WrappedFlowNetwork(NetworkWrapper, FlowNetwork):
    pass
