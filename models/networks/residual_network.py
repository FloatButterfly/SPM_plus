import torch

from models.networks import *
from models.ops import clip, BinaryQuantize
from models.networks.base_network import BaseNetwork

## Base Module ##
class NetModule(nn.Module):
    def __init__(self, in_channels, latent_channels, out_channels):
        super(NetModule, self).__init__()
        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)


## GDN Network input residual, output residual##
class Residual_Encoder_v2(BaseNetwork):
    def __init__(self, in_channels, latent_channels, out_channels):  # 3,96,48
        super(Residual_Encoder_v2, self).__init__()
        print("use residual encoder v2")
        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)
        # self.sin_channel = int(in_channels // 2)
        self.layer_seq1 = nn.Sequential(
            SignConv2d(self._nic, self._nlc, 5, 2, upsample=False, use_bias=True),  # 1,128,5,2
            GDN2d(self._nlc, inverse=False),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            GDN2d(self._nlc, inverse=False),
        )

        self.layer_seq2 = nn.Sequential(
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            GDN2d(self._nlc, inverse=False),
            SignConv2d(self._nlc, self._noc, 5, 2, upsample=False, use_bias=True)
        )

    def forward(self, x_real, x_compressed, vis=False):
        # shape should be [B,3,h,w]
        x = x_real-x_compressed
        x1 = self.layer_seq1(x)
        out = self.layer_seq2(x1)
        # print(out.shape)
        if vis:
            return out,x1
        return out


class Residual_Encoder(BaseNetwork):
    def __init__(self, in_channels, latent_channels, out_channels):  # 6,96,16
        super(Residual_Encoder, self).__init__()

        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)
        self.sin_channel = int(in_channels // 2)
        self.layer_seq1 = nn.Sequential(
            SignConv2d(self._nic, self._nlc, 5, 2, upsample=False, use_bias=True),  # 1,128,5,2
            GDN2d(self._nlc, inverse=False),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            GDN2d(self._nlc, inverse=False),
        )

        self.attention_seq = nn.Sequential(
            SignConv2d(self.sin_channel, self._nlc, 5, 2, upsample=False, use_bias=True),
            nn.ReLU(True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            nn.ReLU(True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            nn.ReLU(True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            nn.Sigmoid(),
        )

        self.layer_seq2 = nn.Sequential(
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            GDN2d(self._nlc, inverse=False),
            SignConv2d(self._nlc, self._noc, 5, 2, upsample=False, use_bias=True)
        )

    def forward(self, x_real, x_compressed, vis=False):
        # shape should be [B,3,h,w]
        x = torch.cat((x_real, x_compressed), dim=1)
        x1 = self.layer_seq1(x)
        att = self.attention_seq(x_compressed)
        # print(x1.shape, att.shape)
        x1 = att * x1
        out = self.layer_seq2(x1)
        # print(out.shape)
        if vis:
            return out, att
        return out



class Residual_Decoder(BaseNetwork):
    def __init__(self, in_channels, latent_channels, out_channels):
        super(Residual_Decoder, self).__init__()
        # 16,96,3

        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)
        self.sin_channel = int(out_channels)
        self.layer_seq1 = nn.Sequential(
            SignConv2d(self._nic, self._nlc, 5, 2, upsample=True, use_bias=True),  # 16,96,5,2
            GDN2d(self._nlc, inverse=True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
        )

        self.attention_seq = nn.Sequential(
            SignConv2d(self.sin_channel, self._nlc, 5, 2, upsample=False, use_bias=True),
            nn.ReLU(True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            nn.ReLU(True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            nn.ReLU(True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=False, use_bias=True),
            nn.Sigmoid(),
        )

        self.layer_seq2 = nn.Sequential(
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
            SignConv2d(self._nlc, self._noc, 5, 2, upsample=True, use_bias=True)
        )

    def forward(self, latent_residual, x_compressed, vis=False):
        x1 = self.layer_seq1(latent_residual)
        att = self.attention_seq(x_compressed)
        # print(x1.shape, att.shape)
        x1 = att * x1
        out = self.layer_seq2(x1)
        # print(out.shape)
        if vis:
            return out, vis
        return out


class Residual_Decoder_v2(BaseNetwork):
    def __init__(self, in_channels, latent_channels, out_channels):
        super(Residual_Decoder_v2, self).__init__()
        # 16,96,3
        print("use residual decoder v2")
        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)
      
        self.layer_seq1 = nn.Sequential(
            SignConv2d(self._nic, self._nlc, 5, 2, upsample=True, use_bias=True),  # 16,96,5,2
            GDN2d(self._nlc, inverse=True),
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
        )

        self.layer_seq2 = nn.Sequential(
            SignConv2d(self._nlc, self._nlc, 5, 2, upsample=True, use_bias=True),
            GDN2d(self._nlc, inverse=True),
            SignConv2d(self._nlc, self._noc, 5, 2, upsample=True, use_bias=True)
        )

    def forward(self, latent_residual, x_compressed, vis=False):
        x1 = self.layer_seq1(latent_residual)
        out = self.layer_seq2(x1)
        # print(out.shape)
        if vis:
            return out, x1
        return out

## GDN Network ##
class Refinement(BaseNetwork):
    def __init__(self, in_channels, latent_channels_1, latent_channels, out_channels):  # 3,48,64,3
        super(Refinement, self).__init__()
        
        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)
        self.nlc1 = int(latent_channels_1)
        self.nlc2 = int(latent_channels - latent_channels_1)

        self.resblock_seq1 = nn.Sequential(
            SignConv2d(self._nic, self.nlc1, 3, 1, upsample=False, use_bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(self.nlc1, self.nlc1, downsample=False),
            ResBlock(self.nlc1, self.nlc1, downsample=False),
        )

        self.resblock_seq2 = nn.Sequential(
            SignConv2d(self._nic, self.nlc2, 3, 1, upsample=False, use_bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(self.nlc2, self.nlc2, downsample=False),
            ResBlock(self.nlc2, self.nlc2, downsample=False),
        )

        self.resblock_seq3 = nn.Sequential(
            ResBlock(self._nlc, self._nlc, downsample=False),
            ResBlock(self._nlc, self._nlc, downsample=False),
            ResBlock(self._nlc, self._nlc, downsample=False),
            SignConv2d(self._nlc, self._noc, 3, 1, upsample=False, use_bias=True),
        )

    def forward(self, decode_residual, x_compressed, vis=False):
        x1 = self.resblock_seq1(x_compressed)
        x2 = self.resblock_seq2(decode_residual)
        # print(x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim=1)
        out = self.resblock_seq3(x)
        if vis:
            return out, x1, x2
        return out


class Refinement_v2(BaseNetwork):
    def __init__(self, in_channels, latent_channels, out_channels):  # 3,64,3
        super(Refinement_v2, self).__init__()
        
        self._nic = int(in_channels)
        self._nlc = int(latent_channels)
        self._noc = int(out_channels)

        self.resblock_seq1 = nn.Sequential(
            SignConv2d(self._nic, self._nlc, 3, 1, upsample=False, use_bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(self._nlc, self._nlc, downsample=False),
            ResBlock(self._nlc, self._nlc, downsample=False),
        )

        
        self.resblock_seq2 = nn.Sequential(
            ResBlock(self._nlc, self._nlc, downsample=False),
            ResBlock(self._nlc, self._nlc, downsample=False),
            ResBlock(self._nlc, self._nlc, downsample=False),
            SignConv2d(self._nlc, self._noc, 3, 1, upsample=False, use_bias=True),
        )

    def forward(self, decode_residual, x_compressed, vis=False):
        # x1 = self.resblock_seq1(x_compressed)
        # x2 = self.resblock_seq2(decode_residual)
        # # print(x1.shape, x2.shape)
        # x = torch.cat((x1, x2), dim=1)
        x1 = x_compressed + decode_residual
        x = self.resblock_seq1(x1)
        out = self.resblock_seq2(x)
        if vis:
            return out, x
        return out


class ResidualEntropyBottleneck(BaseNetwork):
    def __init__(self, in_channels, latent_channels):
        super(ResidualEntropyBottleneck, self).__init__()
        # 16, 96
        self.nic = int(in_channels)  # 16
        self._ch = int(latent_channels)  # 96

        self.hyper_encoder = nn.Sequential(
            SignConv2d(self.nic, self._ch, 5, 2, upsample=False, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self._ch, self._ch, 5, 2, upsample=False, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self._ch, self._ch, 5, 2, upsample=False, use_bias=True)
        )

        self.hyper_decoder = nn.Sequential(
            SignConv2d(self._ch, self._ch, 5, 2, upsample=True, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self._ch, self._ch, 5, 2, upsample=True, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self._ch, self._ch, 5, 2, upsample=True, use_bias=True),
        )

        self.hyper_parameter = nn.Sequential(
            nn.Conv2d(self._ch, self._ch * 2 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch * 2 // 3, self.nic, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.nic, self.nic * 2, 1),
        )

        self.quantizer = Quantizer()
        self.factorized = FullFactorizedModel(self._ch, (3, 3, 3), 1e-9, True)
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        y_hat = self.quantizer(y, self.training)  # 量化

        z = self.hyper_encoder(y)  # 求先验
        z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        u = self.hyper_decoder(z_hat)  # 先验信息解码
        parameters = self.hyper_parameter(u)
        loc, scale = parameters.split(self.nic, dim=1)  # 参数分成 方差，均值
        y_prob, _ = self.conditional(y_hat, loc, scale)  # 高斯模型求概率
        length = torch.sum(-torch.log2(z_prob)) + torch.sum(-torch.log2(y_prob))  # 求出总熵

        return y_hat, z_hat, length

    @property
    def offset(self):
        return self.factorized.integer_offset()


if __name__ == "__main__":
    # import sys
    # print(sys.path)
    residual_encoder = Residual_Encoder(6, 96, 16)
    residual_decoder = Residual_Decoder(16, 96, 3)
    refine_model = Refinement(3, 48, 64, 3)
    residual_entropy_model = ResidualEntropyBottleneck(16, 96)

    x_real = torch.randn((1, 3, 256, 256))
    x_compressed = torch.randn((1, 3, 256, 256))

    latent_residual = residual_encoder(x_real, x_compressed)
    latent_residual_hat, _, bits_length = residual_entropy_model(latent_residual)
    decode_residual = residual_decoder(latent_residual_hat, x_compressed)
    refine_img = refine_model(decode_residual, x_compressed)

    # print(latent_residual.shape, decode_residual.shape, refine_img.shape)
