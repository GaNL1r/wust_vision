import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchaudio
except:
    pass


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv2d_0 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=3,
            kernel_size=(6, 6),
            out_channels=16,
            padding=(2, 2),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_1 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=16,
            in_channels=16,
            kernel_size=(3, 3),
            out_channels=16,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_2 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=16,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_3 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=32,
            in_channels=32,
            kernel_size=(3, 3),
            out_channels=32,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_4 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=16,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_5 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=16,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_6 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=16,
            in_channels=16,
            kernel_size=(3, 3),
            out_channels=16,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_7 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=16,
            kernel_size=(1, 1),
            out_channels=48,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_0 = nn.ChannelShuffle(groups=2)
        self.conv2d_8 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_9 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=32,
            in_channels=32,
            kernel_size=(3, 3),
            out_channels=32,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_10 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_1 = nn.ChannelShuffle(groups=2)
        self.conv2d_11 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_12 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=32,
            in_channels=32,
            kernel_size=(3, 3),
            out_channels=32,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_13 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_2 = nn.ChannelShuffle(groups=2)
        self.conv2d_14 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(3, 3),
            out_channels=64,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_15 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_16 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_17 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=32,
            in_channels=32,
            kernel_size=(3, 3),
            out_channels=32,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_18 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=96,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_3 = nn.ChannelShuffle(groups=2)
        self.conv2d_19 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_20 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(3, 3),
            out_channels=64,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_21 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_4 = nn.ChannelShuffle(groups=2)
        self.conv2d_22 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_23 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(3, 3),
            out_channels=64,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_24 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_5 = nn.ChannelShuffle(groups=2)
        self.conv2d_25 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=128,
            in_channels=128,
            kernel_size=(3, 3),
            out_channels=128,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_26 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_27 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_28 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(3, 3),
            out_channels=64,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_29 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=192,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_6 = nn.ChannelShuffle(groups=2)
        self.conv2d_30 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_31 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=128,
            in_channels=128,
            kernel_size=(3, 3),
            out_channels=128,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_32 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_7 = nn.ChannelShuffle(groups=2)
        self.conv2d_33 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_34 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=128,
            in_channels=128,
            kernel_size=(3, 3),
            out_channels=128,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_35 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.channelshuffle_8 = nn.ChannelShuffle(groups=2)
        self.conv2d_36 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=256,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_37 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_38 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(3, 3),
            out_channels=64,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_39 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_40 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=256,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_41 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=256,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_42 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_43 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(3, 3),
            out_channels=64,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_44 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_45 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_46 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_47 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_48 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_49 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_50 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=32,
            in_channels=32,
            kernel_size=(3, 3),
            out_channels=32,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_51 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=32,
            kernel_size=(1, 1),
            out_channels=32,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_52 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_53 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=128,
            in_channels=128,
            kernel_size=(3, 3),
            out_channels=128,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(2, 2),
        )
        self.conv2d_54 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_55 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=256,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_56 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=256,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_57 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_58 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=128,
            in_channels=128,
            kernel_size=(3, 3),
            out_channels=128,
            padding=(1, 1),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_59 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_60 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=256,
            kernel_size=(1, 1),
            out_channels=256,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_61 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_62 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(5, 5),
            out_channels=64,
            padding=(2, 2),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_63 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=2,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_64 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=128,
            in_channels=128,
            kernel_size=(5, 5),
            out_channels=128,
            padding=(2, 2),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_65 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=2,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_66 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=12,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_67 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=8,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_68 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=1,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_69 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_70 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(5, 5),
            out_channels=64,
            padding=(2, 2),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_71 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=2,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_72 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=128,
            in_channels=128,
            kernel_size=(5, 5),
            out_channels=128,
            padding=(2, 2),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_73 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=2,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_74 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=12,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_75 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=8,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_76 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=1,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_77 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=256,
            kernel_size=(1, 1),
            out_channels=64,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_78 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=64,
            in_channels=64,
            kernel_size=(5, 5),
            out_channels=64,
            padding=(2, 2),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_79 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=2,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_80 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=128,
            in_channels=128,
            kernel_size=(5, 5),
            out_channels=128,
            padding=(2, 2),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_81 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=2,
            in_channels=128,
            kernel_size=(1, 1),
            out_channels=128,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_82 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=12,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_83 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=8,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )
        self.conv2d_84 = nn.Conv2d(
            bias=True,
            dilation=(1, 1),
            groups=1,
            in_channels=64,
            kernel_size=(1, 1),
            out_channels=1,
            padding=(0, 0),
            padding_mode="zeros",
            stride=(1, 1),
        )

        archive = zipfile.ZipFile(
            "/home/hy/wust_vision/model/opt_1208_001.pnnx.bin", "r"
        )
        self.conv2d_0.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_0.bias", (16), "float32"
        )
        self.conv2d_0.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_0.weight", (16, 3, 6, 6), "float32"
        )
        self.conv2d_1.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_1.bias", (16), "float32"
        )
        self.conv2d_1.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_1.weight", (16, 1, 3, 3), "float32"
        )
        self.conv2d_2.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_2.bias", (32), "float32"
        )
        self.conv2d_2.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_2.weight", (32, 16, 1, 1), "float32"
        )
        self.conv2d_3.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_3.bias", (32), "float32"
        )
        self.conv2d_3.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_3.weight", (32, 1, 3, 3), "float32"
        )
        self.conv2d_4.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_4.bias", (16), "float32"
        )
        self.conv2d_4.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_4.weight", (16, 32, 1, 1), "float32"
        )
        self.conv2d_5.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_5.bias", (16), "float32"
        )
        self.conv2d_5.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_5.weight", (16, 32, 1, 1), "float32"
        )
        self.conv2d_6.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_6.bias", (16), "float32"
        )
        self.conv2d_6.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_6.weight", (16, 1, 3, 3), "float32"
        )
        self.conv2d_7.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_7.bias", (48), "float32"
        )
        self.conv2d_7.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_7.weight", (48, 16, 1, 1), "float32"
        )
        self.conv2d_8.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_8.bias", (32), "float32"
        )
        self.conv2d_8.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_8.weight", (32, 32, 1, 1), "float32"
        )
        self.conv2d_9.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_9.bias", (32), "float32"
        )
        self.conv2d_9.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_9.weight", (32, 1, 3, 3), "float32"
        )
        self.conv2d_10.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_10.bias", (32), "float32"
        )
        self.conv2d_10.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_10.weight", (32, 32, 1, 1), "float32"
        )
        self.conv2d_11.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_11.bias", (32), "float32"
        )
        self.conv2d_11.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_11.weight", (32, 32, 1, 1), "float32"
        )
        self.conv2d_12.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_12.bias", (32), "float32"
        )
        self.conv2d_12.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_12.weight", (32, 1, 3, 3), "float32"
        )
        self.conv2d_13.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_13.bias", (32), "float32"
        )
        self.conv2d_13.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_13.weight", (32, 32, 1, 1), "float32"
        )
        self.conv2d_14.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_14.bias", (64), "float32"
        )
        self.conv2d_14.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_14.weight", (64, 1, 3, 3), "float32"
        )
        self.conv2d_15.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_15.bias", (32), "float32"
        )
        self.conv2d_15.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_15.weight", (32, 64, 1, 1), "float32"
        )
        self.conv2d_16.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_16.bias", (32), "float32"
        )
        self.conv2d_16.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_16.weight", (32, 64, 1, 1), "float32"
        )
        self.conv2d_17.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_17.bias", (32), "float32"
        )
        self.conv2d_17.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_17.weight", (32, 1, 3, 3), "float32"
        )
        self.conv2d_18.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_18.bias", (96), "float32"
        )
        self.conv2d_18.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_18.weight", (96, 32, 1, 1), "float32"
        )
        self.conv2d_19.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_19.bias", (64), "float32"
        )
        self.conv2d_19.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_19.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_20.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_20.bias", (64), "float32"
        )
        self.conv2d_20.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_20.weight", (64, 1, 3, 3), "float32"
        )
        self.conv2d_21.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_21.bias", (64), "float32"
        )
        self.conv2d_21.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_21.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_22.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_22.bias", (64), "float32"
        )
        self.conv2d_22.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_22.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_23.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_23.bias", (64), "float32"
        )
        self.conv2d_23.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_23.weight", (64, 1, 3, 3), "float32"
        )
        self.conv2d_24.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_24.bias", (64), "float32"
        )
        self.conv2d_24.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_24.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_25.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_25.bias", (128), "float32"
        )
        self.conv2d_25.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_25.weight", (128, 1, 3, 3), "float32"
        )
        self.conv2d_26.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_26.bias", (64), "float32"
        )
        self.conv2d_26.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_26.weight", (64, 128, 1, 1), "float32"
        )
        self.conv2d_27.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_27.bias", (64), "float32"
        )
        self.conv2d_27.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_27.weight", (64, 128, 1, 1), "float32"
        )
        self.conv2d_28.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_28.bias", (64), "float32"
        )
        self.conv2d_28.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_28.weight", (64, 1, 3, 3), "float32"
        )
        self.conv2d_29.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_29.bias", (192), "float32"
        )
        self.conv2d_29.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_29.weight", (192, 64, 1, 1), "float32"
        )
        self.conv2d_30.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_30.bias", (128), "float32"
        )
        self.conv2d_30.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_30.weight", (128, 128, 1, 1), "float32"
        )
        self.conv2d_31.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_31.bias", (128), "float32"
        )
        self.conv2d_31.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_31.weight", (128, 1, 3, 3), "float32"
        )
        self.conv2d_32.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_32.bias", (128), "float32"
        )
        self.conv2d_32.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_32.weight", (128, 128, 1, 1), "float32"
        )
        self.conv2d_33.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_33.bias", (128), "float32"
        )
        self.conv2d_33.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_33.weight", (128, 128, 1, 1), "float32"
        )
        self.conv2d_34.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_34.bias", (128), "float32"
        )
        self.conv2d_34.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_34.weight", (128, 1, 3, 3), "float32"
        )
        self.conv2d_35.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_35.bias", (128), "float32"
        )
        self.conv2d_35.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_35.weight", (128, 128, 1, 1), "float32"
        )
        self.conv2d_36.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_36.bias", (128), "float32"
        )
        self.conv2d_36.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_36.weight", (128, 256, 1, 1), "float32"
        )
        self.conv2d_37.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_37.bias", (64), "float32"
        )
        self.conv2d_37.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_37.weight", (64, 128, 1, 1), "float32"
        )
        self.conv2d_38.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_38.bias", (64), "float32"
        )
        self.conv2d_38.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_38.weight", (64, 1, 3, 3), "float32"
        )
        self.conv2d_39.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_39.bias", (64), "float32"
        )
        self.conv2d_39.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_39.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_40.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_40.bias", (64), "float32"
        )
        self.conv2d_40.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_40.weight", (64, 256, 1, 1), "float32"
        )
        self.conv2d_41.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_41.bias", (64), "float32"
        )
        self.conv2d_41.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_41.weight", (64, 256, 1, 1), "float32"
        )
        self.conv2d_42.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_42.bias", (64), "float32"
        )
        self.conv2d_42.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_42.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_43.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_43.bias", (64), "float32"
        )
        self.conv2d_43.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_43.weight", (64, 1, 3, 3), "float32"
        )
        self.conv2d_44.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_44.bias", (64), "float32"
        )
        self.conv2d_44.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_44.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_45.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_45.bias", (128), "float32"
        )
        self.conv2d_45.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_45.weight", (128, 128, 1, 1), "float32"
        )
        self.conv2d_46.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_46.bias", (64), "float32"
        )
        self.conv2d_46.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_46.weight", (64, 128, 1, 1), "float32"
        )
        self.conv2d_47.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_47.bias", (32), "float32"
        )
        self.conv2d_47.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_47.weight", (32, 128, 1, 1), "float32"
        )
        self.conv2d_48.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_48.bias", (32), "float32"
        )
        self.conv2d_48.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_48.weight", (32, 128, 1, 1), "float32"
        )
        self.conv2d_49.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_49.bias", (32), "float32"
        )
        self.conv2d_49.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_49.weight", (32, 32, 1, 1), "float32"
        )
        self.conv2d_50.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_50.bias", (32), "float32"
        )
        self.conv2d_50.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_50.weight", (32, 1, 3, 3), "float32"
        )
        self.conv2d_51.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_51.bias", (32), "float32"
        )
        self.conv2d_51.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_51.weight", (32, 32, 1, 1), "float32"
        )
        self.conv2d_52.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_52.bias", (64), "float32"
        )
        self.conv2d_52.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_52.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_53.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_53.bias", (128), "float32"
        )
        self.conv2d_53.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_53.weight", (128, 1, 3, 3), "float32"
        )
        self.conv2d_54.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_54.bias", (128), "float32"
        )
        self.conv2d_54.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_54.weight", (128, 128, 1, 1), "float32"
        )
        self.conv2d_55.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_55.bias", (128), "float32"
        )
        self.conv2d_55.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_55.weight", (128, 256, 1, 1), "float32"
        )
        self.conv2d_56.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_56.bias", (128), "float32"
        )
        self.conv2d_56.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_56.weight", (128, 256, 1, 1), "float32"
        )
        self.conv2d_57.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_57.bias", (128), "float32"
        )
        self.conv2d_57.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_57.weight", (128, 128, 1, 1), "float32"
        )
        self.conv2d_58.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_58.bias", (128), "float32"
        )
        self.conv2d_58.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_58.weight", (128, 1, 3, 3), "float32"
        )
        self.conv2d_59.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_59.bias", (128), "float32"
        )
        self.conv2d_59.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_59.weight", (128, 128, 1, 1), "float32"
        )
        self.conv2d_60.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_60.bias", (256), "float32"
        )
        self.conv2d_60.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_60.weight", (256, 256, 1, 1), "float32"
        )
        self.conv2d_61.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_61.bias", (64), "float32"
        )
        self.conv2d_61.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_61.weight", (64, 64, 1, 1), "float32"
        )
        self.conv2d_62.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_62.bias", (64), "float32"
        )
        self.conv2d_62.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_62.weight", (64, 1, 5, 5), "float32"
        )
        self.conv2d_63.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_63.bias", (128), "float32"
        )
        self.conv2d_63.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_63.weight", (128, 32, 1, 1), "float32"
        )
        self.conv2d_64.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_64.bias", (128), "float32"
        )
        self.conv2d_64.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_64.weight", (128, 1, 5, 5), "float32"
        )
        self.conv2d_65.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_65.bias", (128), "float32"
        )
        self.conv2d_65.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_65.weight", (128, 64, 1, 1), "float32"
        )
        self.conv2d_66.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_66.bias", (12), "float32"
        )
        self.conv2d_66.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_66.weight", (12, 64, 1, 1), "float32"
        )
        self.conv2d_67.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_67.bias", (8), "float32"
        )
        self.conv2d_67.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_67.weight", (8, 64, 1, 1), "float32"
        )
        self.conv2d_68.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_68.bias", (1), "float32"
        )
        self.conv2d_68.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_68.weight", (1, 64, 1, 1), "float32"
        )
        self.conv2d_69.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_69.bias", (64), "float32"
        )
        self.conv2d_69.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_69.weight", (64, 128, 1, 1), "float32"
        )
        self.conv2d_70.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_70.bias", (64), "float32"
        )
        self.conv2d_70.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_70.weight", (64, 1, 5, 5), "float32"
        )
        self.conv2d_71.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_71.bias", (128), "float32"
        )
        self.conv2d_71.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_71.weight", (128, 32, 1, 1), "float32"
        )
        self.conv2d_72.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_72.bias", (128), "float32"
        )
        self.conv2d_72.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_72.weight", (128, 1, 5, 5), "float32"
        )
        self.conv2d_73.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_73.bias", (128), "float32"
        )
        self.conv2d_73.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_73.weight", (128, 64, 1, 1), "float32"
        )
        self.conv2d_74.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_74.bias", (12), "float32"
        )
        self.conv2d_74.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_74.weight", (12, 64, 1, 1), "float32"
        )
        self.conv2d_75.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_75.bias", (8), "float32"
        )
        self.conv2d_75.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_75.weight", (8, 64, 1, 1), "float32"
        )
        self.conv2d_76.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_76.bias", (1), "float32"
        )
        self.conv2d_76.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_76.weight", (1, 64, 1, 1), "float32"
        )
        self.conv2d_77.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_77.bias", (64), "float32"
        )
        self.conv2d_77.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_77.weight", (64, 256, 1, 1), "float32"
        )
        self.conv2d_78.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_78.bias", (64), "float32"
        )
        self.conv2d_78.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_78.weight", (64, 1, 5, 5), "float32"
        )
        self.conv2d_79.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_79.bias", (128), "float32"
        )
        self.conv2d_79.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_79.weight", (128, 32, 1, 1), "float32"
        )
        self.conv2d_80.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_80.bias", (128), "float32"
        )
        self.conv2d_80.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_80.weight", (128, 1, 5, 5), "float32"
        )
        self.conv2d_81.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_81.bias", (128), "float32"
        )
        self.conv2d_81.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_81.weight", (128, 64, 1, 1), "float32"
        )
        self.conv2d_82.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_82.bias", (12), "float32"
        )
        self.conv2d_82.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_82.weight", (12, 64, 1, 1), "float32"
        )
        self.conv2d_83.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_83.bias", (8), "float32"
        )
        self.conv2d_83.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_83.weight", (8, 64, 1, 1), "float32"
        )
        self.conv2d_84.bias = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_84.bias", (1), "float32"
        )
        self.conv2d_84.weight = self.load_pnnx_bin_as_parameter(
            archive, "conv2d_84.weight", (1, 64, 1, 1), "float32"
        )
        archive.close()

    def load_pnnx_bin_as_parameter(
        self, archive, key, shape, dtype, requires_grad=True
    ):
        return nn.Parameter(
            self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad
        )

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode="r", shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.conv2d_0(v_0)
        v_2 = F.hardswish(input=v_1)
        v_3 = self.conv2d_1(v_2)
        v_4 = self.conv2d_2(v_3)
        v_5 = F.hardswish(input=v_4)
        v_6 = self.conv2d_3(v_5)
        v_7 = self.conv2d_4(v_6)
        v_8 = F.hardswish(input=v_7)
        v_9 = self.conv2d_5(v_5)
        v_10 = F.hardswish(input=v_9)
        v_11 = self.conv2d_6(v_10)
        v_12 = self.conv2d_7(v_11)
        v_13 = F.hardswish(input=v_12)
        v_14 = torch.cat((v_8, v_13), dim=1)
        v_15 = self.channelshuffle_0(v_14)
        v_16, v_17 = torch.tensor_split(v_15, dim=1, indices=(32,))
        v_18 = self.conv2d_8(v_17)
        v_19 = F.hardswish(input=v_18)
        v_20 = self.conv2d_9(v_19)
        v_21 = self.conv2d_10(v_20)
        v_22 = F.hardswish(input=v_21)
        v_23 = torch.cat((v_16, v_22), dim=1)
        v_24 = self.channelshuffle_1(v_23)
        v_25, v_26 = torch.tensor_split(v_24, dim=1, indices=(32,))
        v_27 = self.conv2d_11(v_26)
        v_28 = F.hardswish(input=v_27)
        v_29 = self.conv2d_12(v_28)
        v_30 = self.conv2d_13(v_29)
        v_31 = F.hardswish(input=v_30)
        v_32 = torch.cat((v_25, v_31), dim=1)
        v_33 = self.channelshuffle_2(v_32)
        v_34 = self.conv2d_14(v_33)
        v_35 = self.conv2d_15(v_34)
        v_36 = F.hardswish(input=v_35)
        v_37 = self.conv2d_16(v_33)
        v_38 = F.hardswish(input=v_37)
        v_39 = self.conv2d_17(v_38)
        v_40 = self.conv2d_18(v_39)
        v_41 = F.hardswish(input=v_40)
        v_42 = torch.cat((v_36, v_41), dim=1)
        v_43 = self.channelshuffle_3(v_42)
        v_44, v_45 = torch.tensor_split(v_43, dim=1, indices=(64,))
        v_46 = self.conv2d_19(v_45)
        v_47 = F.hardswish(input=v_46)
        v_48 = self.conv2d_20(v_47)
        v_49 = self.conv2d_21(v_48)
        v_50 = F.hardswish(input=v_49)
        v_51 = torch.cat((v_44, v_50), dim=1)
        v_52 = self.channelshuffle_4(v_51)
        v_53, v_54 = torch.tensor_split(v_52, dim=1, indices=(64,))
        v_55 = self.conv2d_22(v_54)
        v_56 = F.hardswish(input=v_55)
        v_57 = self.conv2d_23(v_56)
        v_58 = self.conv2d_24(v_57)
        v_59 = F.hardswish(input=v_58)
        v_60 = torch.cat((v_53, v_59), dim=1)
        v_61 = self.channelshuffle_5(v_60)
        v_62 = self.conv2d_25(v_61)
        v_63 = self.conv2d_26(v_62)
        v_64 = F.hardswish(input=v_63)
        v_65 = self.conv2d_27(v_61)
        v_66 = F.hardswish(input=v_65)
        v_67 = self.conv2d_28(v_66)
        v_68 = self.conv2d_29(v_67)
        v_69 = F.hardswish(input=v_68)
        v_70 = torch.cat((v_64, v_69), dim=1)
        v_71 = self.channelshuffle_6(v_70)
        v_72, v_73 = torch.tensor_split(v_71, dim=1, indices=(128,))
        v_74 = self.conv2d_30(v_73)
        v_75 = F.hardswish(input=v_74)
        v_76 = self.conv2d_31(v_75)
        v_77 = self.conv2d_32(v_76)
        v_78 = F.hardswish(input=v_77)
        v_79 = torch.cat((v_72, v_78), dim=1)
        v_80 = self.channelshuffle_7(v_79)
        v_81, v_82 = torch.tensor_split(v_80, dim=1, indices=(128,))
        v_83 = self.conv2d_33(v_82)
        v_84 = F.hardswish(input=v_83)
        v_85 = self.conv2d_34(v_84)
        v_86 = self.conv2d_35(v_85)
        v_87 = F.hardswish(input=v_86)
        v_88 = torch.cat((v_81, v_87), dim=1)
        v_89 = self.channelshuffle_8(v_88)
        v_90 = self.conv2d_36(v_89)
        v_91 = F.hardswish(input=v_90)
        v_92 = self.conv2d_37(v_91)
        v_93 = F.hardswish(input=v_92)
        v_94 = F.interpolate(
            input=v_93,
            mode="nearest",
            recompute_scale_factor=False,
            scale_factor=(2.0, 2.0),
        )
        v_95 = self.conv2d_38(v_33)
        v_96 = self.conv2d_39(v_95)
        v_97 = F.hardswish(input=v_96)
        v_98 = torch.cat((v_97, v_61, v_94), dim=1)
        v_99 = self.conv2d_40(v_98)
        v_100 = F.hardswish(input=v_99)
        v_101 = self.conv2d_41(v_98)
        v_102 = F.hardswish(input=v_101)
        v_103 = self.conv2d_42(v_100)
        v_104 = F.hardswish(input=v_103)
        v_105 = self.conv2d_43(v_104)
        v_106 = self.conv2d_44(v_105)
        v_107 = F.hardswish(input=v_106)
        v_108 = torch.cat((v_107, v_102), dim=1)
        v_109 = self.conv2d_45(v_108)
        v_110 = F.hardswish(input=v_109)
        v_111 = self.conv2d_46(v_110)
        v_112 = F.hardswish(input=v_111)
        v_113 = F.interpolate(
            input=v_112,
            mode="nearest",
            recompute_scale_factor=False,
            scale_factor=(2.0, 2.0),
        )
        v_114 = torch.cat((v_33, v_113), dim=1)
        v_115 = self.conv2d_47(v_114)
        v_116 = F.hardswish(input=v_115)
        v_117 = self.conv2d_48(v_114)
        v_118 = F.hardswish(input=v_117)
        v_119 = self.conv2d_49(v_116)
        v_120 = F.hardswish(input=v_119)
        v_121 = self.conv2d_50(v_120)
        v_122 = self.conv2d_51(v_121)
        v_123 = F.hardswish(input=v_122)
        v_124 = torch.cat((v_123, v_118), dim=1)
        v_125 = self.conv2d_52(v_124)
        v_126 = F.hardswish(input=v_125)
        v_127 = self.conv2d_53(v_110)
        v_128 = self.conv2d_54(v_127)
        v_129 = F.hardswish(input=v_128)
        v_130 = torch.cat((v_129, v_91), dim=1)
        v_131 = self.conv2d_55(v_130)
        v_132 = F.hardswish(input=v_131)
        v_133 = self.conv2d_56(v_130)
        v_134 = F.hardswish(input=v_133)
        v_135 = self.conv2d_57(v_132)
        v_136 = F.hardswish(input=v_135)
        v_137 = self.conv2d_58(v_136)
        v_138 = self.conv2d_59(v_137)
        v_139 = F.hardswish(input=v_138)
        v_140 = torch.cat((v_139, v_134), dim=1)
        v_141 = self.conv2d_60(v_140)
        v_142 = F.hardswish(input=v_141)
        v_143 = self.conv2d_61(v_126)
        v_144 = F.hardswish(input=v_143)
        v_145 = self.conv2d_62(v_144)
        v_146 = self.conv2d_63(v_145)
        v_147 = F.hardswish(input=v_146)
        v_148 = self.conv2d_64(v_147)
        v_149 = self.conv2d_65(v_148)
        v_150 = F.hardswish(input=v_149)
        v_151, v_152 = torch.tensor_split(v_150, dim=1, indices=(64,))
        v_153 = self.conv2d_66(v_152)
        v_154 = self.conv2d_67(v_151)
        v_155 = self.conv2d_68(v_151)
        v_156 = F.sigmoid(input=v_155)
        v_157 = F.sigmoid(input=v_153)
        v_158 = torch.cat((v_154, v_156, v_157), dim=1)
        v_159 = self.conv2d_69(v_110)
        v_160 = F.hardswish(input=v_159)
        v_161 = self.conv2d_70(v_160)
        v_162 = self.conv2d_71(v_161)
        v_163 = F.hardswish(input=v_162)
        v_164 = self.conv2d_72(v_163)
        v_165 = self.conv2d_73(v_164)
        v_166 = F.hardswish(input=v_165)
        v_167, v_168 = torch.tensor_split(v_166, dim=1, indices=(64,))
        v_169 = self.conv2d_74(v_168)
        v_170 = self.conv2d_75(v_167)
        v_171 = self.conv2d_76(v_167)
        v_172 = F.sigmoid(input=v_171)
        v_173 = F.sigmoid(input=v_169)
        v_174 = torch.cat((v_170, v_172, v_173), dim=1)
        v_175 = self.conv2d_77(v_142)
        v_176 = F.hardswish(input=v_175)
        v_177 = self.conv2d_78(v_176)
        v_178 = self.conv2d_79(v_177)
        v_179 = F.hardswish(input=v_178)
        v_180 = self.conv2d_80(v_179)
        v_181 = self.conv2d_81(v_180)
        v_182 = F.hardswish(input=v_181)
        v_183, v_184 = torch.tensor_split(v_182, dim=1, indices=(64,))
        v_185 = self.conv2d_82(v_184)
        v_186 = self.conv2d_83(v_183)
        v_187 = self.conv2d_84(v_183)
        v_188 = F.sigmoid(input=v_187)
        v_189 = F.sigmoid(input=v_185)
        v_190 = torch.cat((v_186, v_188, v_189), dim=1)
        v_191 = v_158.reshape(1, 21, 2704)
        v_192 = v_174.reshape(1, 21, 676)
        v_193 = v_190.reshape(1, 21, 169)
        v_194 = torch.cat((v_191, v_192, v_193), dim=2)
        v_195 = v_194.permute(dims=(0, 2, 1))
        return v_195


def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 416, 416, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("/home/hy/wust_vision/model/opt_1208_001_pnnx.py.pt")


def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 416, 416, dtype=torch.float)

    torch.onnx.export(
        net,
        v_0,
        "/home/hy/wust_vision/model/opt_1208_001_pnnx.py.onnx",
        export_params=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        opset_version=13,
        input_names=["in0"],
        output_names=["out0"],
    )


@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 416, 416, dtype=torch.float)

    return net(v_0)


if __name__ == "__main__":
    print(test_inference())
