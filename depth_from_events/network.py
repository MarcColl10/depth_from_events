import torch.nn as nn

from depth_from_events.blocks import conv_encoder, flatten_decoder, LazyConvGru, LazyConvMinGru, upsample_decoder
from depth_from_events.network_utils import NetworkWrapper


class FlowNetwork(nn.Module):
    """
    Optical flow prediction network following ID-Net.
    """

    def __init__(
        self,
        mode,
        encoder_channels,
        memory_channels,
        decoder_channels,
        activation_fn,
        final_bias,
        padding_mode,
        scaling,
    ):
        super().__init__()

        self.mode = mode
        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels, activation_fn, padding_mode=padding_mode)
        self.memory = LazyConvGru(memory_channels, 3, padding_mode=padding_mode)
        # self.memory = LazyConvMinGru(memory_channels, 3, padding_mode=padding_mode)
        self.decoder = upsample_decoder(
            decoder_channels, activation_fn, final_bias, padding_mode=padding_mode, mode=self.mode
        )

    def forward(self, input, hidden=None):
        encoder = self.encoder(input[:, :2])  # only polarity channels
        memory = self.memory(encoder, hidden)
        flow_map = self.decoder(memory)

        flow_map *= self.scaling

        return flow_map, memory


class WrappedFlowNetwork(NetworkWrapper, FlowNetwork):
    pass


class DepthPoseNetwork(nn.Module):
    """
    Depth/disparity and pose prediction network following ID-Net.
    """

    def __init__(
        self,
        mode,
        encoder_channels,
        memory_channels,
        decoder_channels,
        activation_fn,
        final_bias,
        padding_mode,
        scaling,
    ):
        super().__init__()

        self.mode = mode
        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels, activation_fn, padding_mode=padding_mode)
        self.memory = LazyConvGru(memory_channels, 3, padding_mode=padding_mode)
        # self.memory = LazyConvMinGru(memory_channels, 3, padding_mode=padding_mode)
        self.depth_decoder = upsample_decoder(
            decoder_channels, activation_fn, final_bias, padding_mode=padding_mode, mode=self.mode
        )
        self.pose_decoder = flatten_decoder(
            decoder_channels, activation_fn, final_bias, padding_mode=padding_mode, mode="pose"
        )

    def forward(self, input, hidden=None):
        encoder = self.encoder(input[:, :2])  # only polarity channels
        memory = self.memory(encoder, hidden)
        depth_map = self.depth_decoder(memory)
        pose = self.pose_decoder(memory)

        depth_scale, pose_scale = self.scaling
        depth_map *= depth_scale
        pose *= pose_scale

        return (depth_map, pose), memory


class WrappedDepthPoseNetwork(NetworkWrapper, DepthPoseNetwork):
    pass


class DepthPoseIntrinsicsNetwork(nn.Module):
    """
    Depth/disparity, pose and camera intrinsics prediction network following ID-Net.
    """

    def __init__(self, mode, encoder_channels, memory_channels, decoder_channels, activation_fn, final_bias, scaling):
        super().__init__()

        self.mode = mode
        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels, activation_fn)
        self.memory = LazyConvGru(memory_channels, 3)
        self.depth_decoder = upsample_decoder(decoder_channels, activation_fn, final_bias, mode=self.mode)
        self.pose_decoder = flatten_decoder(decoder_channels, activation_fn, final_bias, mode="pose")
        self.intrinsics_decoder = flatten_decoder(decoder_channels, activation_fn, final_bias, mode="intrinsics")

    def forward(self, input, hidden=None):
        encoder = self.encoder(input[:, :2])  # only polarity channels
        memory = self.memory(encoder, hidden)
        depth_map = self.depth_decoder(memory)
        pose = self.pose_decoder(memory)
        intrinsics = self.intrinsics_decoder(memory)

        depth_scale, pose_scale, intrinsics_scale = self.scaling
        depth_map *= depth_scale
        pose *= pose_scale
        intrinsics *= intrinsics_scale

        return (depth_map, pose, intrinsics), memory


class WrappedDepthPoseIntrinsicsNetwork(NetworkWrapper, DepthPoseIntrinsicsNetwork):
    pass


# temporary aliases
class DisparityPoseNetwork(DepthPoseNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__("depth", *args, **kwargs)


class WrappedDisparityPoseNetwork(NetworkWrapper, DisparityPoseNetwork):
    pass
