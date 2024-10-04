import torch.nn as nn

from blocks import conv_encoder, flatten_decoder, LazyConvGru, upsample_decoder
from network_utils import NetworkWrapper


class FlowNetwork(nn.Module):
    """
    Optical flow prediction network following ID-Net.
    """

    def __init__(self, encoder_channels, memory_channels, decoder_channels, activation_fn, final_bias, scaling):
        super().__init__()

        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels, activation_fn)
        self.memory = LazyConvGru(memory_channels, 3)
        self.decoder = upsample_decoder(decoder_channels, activation_fn, final_bias, mode="flow")

    def forward(self, input, hidden=None):
        encoder = self.encoder(input[:, :2])  # only polarity channels
        memory = self.memory(encoder, hidden)
        flow_map = self.decoder(memory)

        flow_map *= self.scaling

        return flow_map, memory


class WrappedFlowNetwork(NetworkWrapper, FlowNetwork):
    pass


class DisparityPoseNetwork(nn.Module):
    """
    Disparity and pose prediction network following ID-Net.
    """

    def __init__(
        self, encoder_channels, memory_channels, decoder_channels, activation_fn, final_bias, padding_mode, scaling
    ):
        super().__init__()

        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels, activation_fn, padding_mode=padding_mode)
        self.memory = LazyConvGru(memory_channels, 3)
        self.disp_decoder = upsample_decoder(
            decoder_channels, activation_fn, final_bias, padding_mode=padding_mode, mode="disparity"
        )
        self.pose_decoder = flatten_decoder(
            decoder_channels, activation_fn, final_bias, padding_mode=padding_mode, mode="pose"
        )

    def forward(self, input, hidden=None):
        encoder = self.encoder(input[:, :2])  # only polarity channels
        memory = self.memory(encoder, hidden)
        disp_map = self.disp_decoder(memory)
        pose = self.pose_decoder(memory)

        disp_scale, pose_scale = self.scaling
        disp_map *= disp_scale
        pose *= pose_scale

        return (disp_map, pose), memory


class WrappedDisparityPoseNetwork(NetworkWrapper, DisparityPoseNetwork):
    pass


class DisparityPoseIntrinsicsNetwork(nn.Module):
    """
    Disparity, pose and camera intrinsics prediction network following ID-Net.
    """

    modality = "disparity"

    def __init__(self, encoder_channels, memory_channels, decoder_channels, activation_fn, final_bias, scaling):
        super().__init__()

        self.scaling = scaling

        self.encoder = conv_encoder(encoder_channels, activation_fn)
        self.memory = LazyConvGru(memory_channels, 3)
        self.disp_decoder = upsample_decoder(decoder_channels, activation_fn, final_bias, mode="disparity")
        self.pose_decoder = flatten_decoder(decoder_channels, activation_fn, final_bias, mode="pose")
        self.intrinsics_decoder = flatten_decoder(decoder_channels, activation_fn, final_bias, mode="intrinsics")

    def forward(self, input, hidden=None):
        encoder = self.encoder(input[:, :2])  # only polarity channels
        memory = self.memory(encoder, hidden)
        disp_map = self.disp_decoder(memory)
        pose = self.pose_decoder(memory)
        intrinsics = self.intrinsics_decoder(memory)

        disp_scale, pose_scale, intrinsics_scale = self.scaling
        disp_map *= disp_scale
        pose *= pose_scale
        intrinsics *= intrinsics_scale

        return (disp_map, pose, intrinsics), memory


class WrappedDisparityPoseIntrinsicsNetwork(NetworkWrapper, DisparityPoseIntrinsicsNetwork):
    pass
