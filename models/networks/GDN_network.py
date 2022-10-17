from .gdn import GDN2d
from .sign_conv2d import SignConv2d
from ..networks import *


## Base Module ##
class NetModule(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super(NetModule, self).__init__()
        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)


## GDN Network ##
class Encoder_GDN(NetModule):
    def __init__(self, in_channels, latent_channels, out_channels):  # 1,128,128
        super(Encoder_GDN, self).__init__(in_channels, latent_channels, out_channels)

        self.model = nn.Sequential(
            SignConv2d(self._nic, self._nlc, 5, 2, upsample=False, use_bias=True),  # 1,128,5,2
            GDN2d(self._nlc, inverse=False),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            GDN2d(self._nlc, inverse=False),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            GDN2d(self._nlc, inverse=False),
            SignConv2d(self._nlc, self._noc, 5, 2, upsample=False, use_bias=True)
        )

    def forward(self, x):
        return self.model(x)


class Decoder_GDN(NetModule):
    def __init__(self, in_channels, latent_channels, out_channels):
        super(Decoder_GDN, self).__init__(in_channels, latent_channels, out_channels)

        self.model = nn.Sequential(
            SignConv2d(self._nic, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
            SignConv2d(self._nlc, self._noc, 5, 2, upsample=True, use_bias=True)
        )

    def forward(self, x):
        return self.model(x)
