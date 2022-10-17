# -*- coding: utf-8 -*-
from .networks import *
from .ops import clip, BinaryQuantize


class EntropyBottleneck(nn.Module):
    def __init__(self, channels):
        super(EntropyBottleneck, self).__init__()
        self._ch = int(channels)

        self.hyper_encoder = nn.Sequential(
            SignConv2d(self._ch, self._ch, 3, 1, upsample=False, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self._ch, self._ch, 5, 2, upsample=False, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self._ch, self._ch, 5, 2, upsample=False, use_bias=True)
        )

        self.hyper_decoder = nn.Sequential(
            SignConv2d(self._ch, self._ch, 5, 2, upsample=True, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self._ch, self._ch * 3 // 2, 5, 2, upsample=True, use_bias=True),
            nn.LeakyReLU(inplace=True),
            SignConv2d(self._ch * 3 // 2, self._ch * 2, 3, 1, upsample=True, use_bias=True)
        )

        self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        self.hyper_parameter = nn.Sequential(
            nn.Conv2d(self._ch * 4, self._ch * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch * 10 // 3, self._ch * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch * 8 // 3, self._ch * 2, 1)
        )

        self.quantizer = Quantizer()
        self.factorized = FullFactorizedModel(self._ch, (3, 3, 3), 1e-9, True)
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        y_hat = self.quantizer(y, self.training)  # 量化

        z = self.hyper_encoder(y)  # 求先验
        z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        u = self.hyper_decoder(z_hat)  # 先验信息解码
        v = self.context_model(y_hat)  # 上下文模型卷积
        # if u.shape != v.shape:
        #     print(z_hat.shape, u.shape, v.shape, y.shape)
        parameters = self.hyper_parameter(torch.cat((u, v), dim=1))  # 上下文与先验融合求概率分布参数
        loc, scale = parameters.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob = self.conditional(y_hat, loc, scale)  # 高斯模型求概率

        length = torch.sum(-torch.log2(z_prob)) + torch.sum(-torch.log2(y_prob))  # 求出总熵

        return y_hat, z_hat, length

    @property
    def offset(self):
        return self.factorized.integer_offset()


# For ablation study: use factorized prior model
class TextureEntropyBottleneck_no_hyperprior(BaseNetwork):
    def __init__(self, channels):
        super(TextureEntropyBottleneck_no_hyperprior, self).__init__()
        super().__init__()
        self._ch = int(channels)
#         print(channels)
        self.quantizer = Quantizer()
        self.factorized = FullFactorizedModel(self._ch, (3, 3, 3), 1e-9, True)
        # self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        # y_hat = self.quantizer(y, self.training)  # 量化

        # z = self.hyper_encoder(y)  # 求先验 [bs,16,19,1]
        y_hat, y_prob = self.factorized(y)  # independent probability estimation

        length = torch.sum(-torch.log2(y_prob))  # 求出总熵

        return y_hat, y_hat, length, y_prob

    @property
    def offset(self):
        return self.factorized.integer_offset()


# Cross-channel entropy model, without adap qp networks in journal
class TextureEntropyBottleneck(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck, self).__init__()
        self._ch = int(channels)
        self.opt = opt
        if self._ch < 32:  # channels must >= 8
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 8, 1, 1)
            )

            self.hyper_decoder = nn.Sequential(
                nn.Conv2d(self._ch // 8, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
            )
            self.factorized = FullFactorizedModel(self._ch // 8, (3, 3, 3), 1e-9, True)
        else:
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
            )

            self.hyper_decoder = nn.Sequential(
                nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
            )

            if opt.adap_qp:
                if not opt.adap_qp_region_wise:
                    self.adap_qp_net = nn.Sequential(
                        nn.Conv2d(self._ch * 2, self._ch * 2, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self._ch * 2, self._ch * 2, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self._ch * 2, self._ch, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        # nn.Sigmoid()
                    )
                else:
                    self.adap_qp_net = nn.Sequential(
                        nn.Conv2d(self._ch * 2, self._ch // 2, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self._ch // 2, self._ch // 4, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self._ch // 4, 1, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        # nn.Sigmoid()
                    )
                self.linear = nn.Linear(opt.label_nc + 1, opt.label_nc + 1)
            self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        # self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        # self.hyper_parameter = nn.Sequential(
        #     nn.Conv2d(self._ch * 4, self._ch * 10 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 10 // 3, self._ch * 8 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 8 // 3, self._ch * 2, 1)
        # )

        self.quantizer = Quantizer()

        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        z = self.hyper_encoder(y)  # 求先验 [bs,16,19,1]
        z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        u = self.hyper_decoder(z_hat)  # 先验信息解码
        if self.opt.adap_qp:
            # if not self.opt.adap_qp_region_wise:
            #     noise_scale_ori = torch.exp(self.adap_qp_net(u))
            #     noise_scale = clip(noise_scale_ori, self.opt.qp_step, 10)
            # # print("the mean of noise scale: ", noise_scale)
            # # if torch.isnan(noise_scale).any():
            # #     print(noise_scale)
            # #     raise AssertionError("nan in noise scale")
            # else:
            noise_scale_ori = self.adap_qp_net(u)
            #             ori_scale = ori_scale.view(ori_scale.size(0), ori_scale.size(1), ori_scale.size(3), ori_scale.size(2))
            #             noise_scale_ori = self.linear(ori_scale)
            noise_scale_ori = torch.exp(noise_scale_ori)
            #             noise_scale_ori = noise_scale_ori.view(ori_scale.size(0), ori_scale.size(1), ori_scale.size(3), ori_scale.size(2))
            upper_bound = torch.tensor(self.opt.upper_bound)
            lower_bound = torch.tensor(self.opt.lower_bound)
            noise_scale = clip(noise_scale_ori, lower_bound, upper_bound)
            y_hat = self.quantizer(y, self.training, noise_scale=noise_scale)  # 量化
        else:
            y_hat = self.quantizer(y, self.training)  # 量化
        # v = self.context_model(y_hat)  # 上下文模型卷积
        # parameters = self.hyper_parameter(torch.cat((u, v), dim=1))  # 上下文与先验融合求概率分布参数
        loc, scale = u.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob, y_decor = self.conditional(y_hat, loc, scale)  # 高斯模型求概率
        length = torch.sum(-torch.log2(z_prob)) + torch.sum(-torch.log2(y_prob))  # 求出总熵
        if self.opt.adap_qp:
            y_hat = y_hat * noise_scale
            return y_hat, z_hat, length, noise_scale
        return y_hat, z_hat, length, y_prob

    @property
    def offset(self):
        return self.factorized.integer_offset()


# TextureEntropyBottleneck_GSM
# Cross-channel entropy model, without adap qp networks in journal
class TextureEntropyBottleneck_GSM(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck_GSM, self).__init__()
        self._ch = int(channels)
        self.opt = opt
        if self._ch < 32:  # channels must >= 8
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 8, 1, 1)
            )

            self.hyper_decoder = nn.Sequential(
                nn.Conv2d(self._ch // 8, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch, 1, 1)
            )
            self.factorized = FullFactorizedModel(self._ch // 8, (3, 3, 3), 1e-9, True)
        else:
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
            )

            self.hyper_decoder = nn.Sequential(
                nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch, 1, 1)
            )

            if opt.adap_qp:
                if not opt.adap_qp_region_wise:
                    self.adap_qp_net = nn.Sequential(
                        nn.Conv2d(self._ch * 2, self._ch * 2, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self._ch * 2, self._ch * 2, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self._ch * 2, self._ch, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        # nn.Sigmoid()
                    )
                else:
                    self.adap_qp_net = nn.Sequential(
                        nn.Conv2d(self._ch * 2, self._ch // 2, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self._ch // 2, self._ch // 4, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self._ch // 4, 1, 1, 1),
                        nn.LeakyReLU(inplace=True),
                        # nn.Sigmoid()
                    )
                self.linear = nn.Linear(opt.label_nc + 1, opt.label_nc + 1)
            self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        # self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        # self.hyper_parameter = nn.Sequential(
        #     nn.Conv2d(self._ch * 4, self._ch * 10 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 10 // 3, self._ch * 8 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 8 // 3, self._ch * 2, 1)
        # )

        self.quantizer = Quantizer()

        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        z = self.hyper_encoder(y)  # 求先验 [bs,16,19,1]
        z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        u = self.hyper_decoder(z_hat)  # 先验信息解码
        y_hat = self.quantizer(y, self.training)  # 量化
        loc = torch.zeros_like(u)
        scale = u
#         loc, scale = u.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob, y_decor = self.conditional(y_hat, loc, scale)  # 高斯模型求概率
        length = torch.sum(-torch.log2(z_prob)) + torch.sum(-torch.log2(y_prob))  # 求出总熵
        return y_hat, z_hat, length, y_prob

    @property
    def offset(self):
        return self.factorized.integer_offset()


# Cross-channel entropy model, region adaptive qp mask networks
class TextureEntropyBottleneck_v0(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck_v0, self).__init__()
        self._ch = int(channels)
        self.opt = opt
        if self._ch < 32:  # channels must >= 8
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 8, 1, 1)
            )

            self.hyper_decoder = nn.Sequential(
                nn.Conv2d(self._ch // 8, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
            )
            self.factorized = FullFactorizedModel(self._ch // 8, (3, 3, 3), 1e-9, True)
        else:
            self.hyper_encoder = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
            )

            self.hyper_decoder = nn.Sequential(
                nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
            )
            self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        # self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        # self.hyper_parameter = nn.Sequential(
        #     nn.Conv2d(self._ch * 4, self._ch * 10 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 10 // 3, self._ch * 8 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 8 // 3, self._ch * 2, 1)
        # )
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.Phi = nn.Parameter(init.xavier_normal_(torch.empty((self.opt.label_nc + 1, 1))))
        self.register_parameter('Mask_Phi', self.Phi)
        self.quantizer = Quantizer()
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        mask_matrix = BinaryQuantize(self.Phi, self.k, self.t)
        mask = mask_matrix.unsqueeze(0).unsqueeze(0)
        z = self.hyper_encoder(y)  # 求先验 [bs,16,19,1]
        z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        u = self.hyper_decoder(z_hat)  # 先验信息解码
        if self.opt.adap_qp:
            scaled_y = mask * (y / 0.01) + (1 - mask) * (y / 0.1)
            y_hat = self.quantizer(scaled_y, self.training)  # 量化
            # print(y_hat)
        else:
            y_hat = self.quantizer(y, self.training)  # 量化
        # v = self.context_model(y_hat)  # 上下文模型卷积
        # parameters = self.hyper_parameter(torch.cat((u, v), dim=1))  # 上下文与先验融合求概率分布参数
        loc, scale = u.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob, y_decor = self.conditional(y_hat, loc, scale)  # 高斯模型求概率
        length = torch.sum(-torch.log2(z_prob)) + torch.sum(-torch.log2(y_prob))  # 求出总熵
        if self.opt.adap_qp:
            y_hat = mask * (y_hat * 0.01) + (1 - mask) * (y_hat * 0.1)
            error = torch.mean(torch.abs(y - y_hat))
            # print(y_hat)
            return y_hat, z_hat, length, mask
        return y_hat, z_hat, length, y_prob

    @property
    def offset(self):
        return self.factorized.integer_offset()


# cross-channel + cross region entropy model: parallel
class TextureEntropyBottleneck_v1(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck_v1, self).__init__()
        self._ch = int(channels)
        self.s_ch = opt.label_nc + 1
        self.opt = opt

        # self.factorized_0 = FullFactorizedModel(self._ch // 4, (3, 3, 3), 1e-9, True)
        if self.s_ch > 32:
            self.cross_region_encoder_0 = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.cross_region_encoder = nn.Sequential(
                nn.Conv2d(self.s_ch, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 16, self.s_ch // 32, 1, 1)
            )
            self.cross_region_decoder = nn.Sequential(
                nn.Conv2d(self.s_ch // 32, self.s_ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 16, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.cross_region_decoder_0 = nn.Sequential(
                nn.Conv2d(self._ch // 4, self._ch, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.cross_region_encoder_0 = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.cross_region_encoder = nn.Sequential(
                nn.Conv2d(self.s_ch, self.s_ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 2, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 8, 1, 1)
            )
            self.cross_region_decoder = nn.Sequential(
                nn.Conv2d(self.s_ch // 8, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 2, self.s_ch, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.cross_region_decoder_0 = nn.Sequential(
                nn.Conv2d(self._ch // 4, self._ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self._ch // 2, self._ch, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
        self.cross_region_parameters = nn.Conv2d(self._ch, self._ch * 2, 1, 1)

        self.cross_channel_encoder = nn.Sequential(
            nn.Conv2d(self._ch, self._ch // 4, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
        )

        self.cross_channel_decoder = nn.Sequential(
            nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
        )

        self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        # self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        self.hyper_parameter = nn.Sequential(
            nn.Conv2d(self._ch * 4, self._ch * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch * 10 // 3, self._ch * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch * 8 // 3, self._ch * 2, 1)
        )

        self.quantizer = Quantizer()
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        z_0 = self.cross_region_encoder_0(y)
        z_0 = self.cross_region_encoder(torch.transpose(z_0, 1, 2))
        z_0 = torch.transpose(z_0, 1, 2)
        z0_hat, z0_prob = self.factorized(z_0)
        u_0 = self.cross_region_decoder(torch.transpose(z0_hat, 1, 2))
        u_0 = self.cross_region_decoder_0(torch.transpose(u_0, 1, 2))
        u_0 = self.cross_region_parameters(u_0)

        z = self.cross_channel_encoder(y)  # 求先验 [bs,16,19,1]
        z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        u = self.cross_channel_decoder(z_hat)  # 先验信息解码

        y_hat = self.quantizer(y, self.training)  # 量化
        # v = self.context_model(y_hat)  # 上下文模型卷积
        u = self.hyper_parameter(torch.cat((u_0, u), dim=1))  # 上下文与先验融合求概率分布参数

        loc, scale = u.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob, y_decor = self.conditional(y_hat, loc, scale)  # 高斯模型求概率

        len_z0 = torch.sum(-torch.log2(z0_prob))
        len_z = torch.sum(-torch.log2(z_prob))
        len_y = torch.sum(-torch.log2(y_prob))

        length = len_z0 + len_z + len_y  # 求出总熵

        return y_hat, z_hat, length, y_prob

    @property
    def offset(self):
        return self.factorized.integer_offset()


# cross-channel + cross region entropy model: cascade
class TextureEntropyBottleneck_v2(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck_v2, self).__init__()
        self._ch = int(channels)
        self.s_ch = opt.label_nc + 1
        self.opt = opt

        self.cross_region_encoder = nn.Sequential(
            nn.Conv2d(self.s_ch, self.s_ch, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.s_ch, self.s_ch, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.s_ch, self.s_ch, 1, 1)
        )
        self.cross_channel_encoder = nn.Sequential(
            nn.Conv2d(self._ch, self._ch // 4, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
        )
        self.cross_channel_decoder = nn.Sequential(
            nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
        )
        self.cross_region_decoder = nn.Sequential(
            nn.Conv2d(self.s_ch, self.s_ch, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.s_ch, self.s_ch, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.s_ch, self.s_ch, 1, 1)
        )

        self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        # self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        self.quantizer = Quantizer()
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        z_0 = self.cross_region_encoder(torch.transpose(y, 1, 2))
        z = self.cross_channel_encoder(torch.transpose(z_0, 1, 2))
        z_hat, z_prob = self.factorized(z)
        u_0 = self.cross_channel_decoder(z_hat)
        u = self.cross_region_decoder(torch.transpose(u_0, 1, 2))
        u = torch.transpose(u, 1, 2)
        y_hat = self.quantizer(y, self.training)  # 量化
        # v = self.context_model(y_hat)  # 上下文模型卷积
        loc, scale = u.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob, y_decor = self.conditional(y_hat, loc, scale)  # 高斯模型求概率
        length = torch.sum(-torch.log2(z_prob)) + torch.sum(-torch.log2(y_prob))  # 求出总熵

        return y_hat, z_hat, length, y_prob

    @property
    def offset(self):
        return self.factorized.integer_offset()


# Ablation study: only cross region entropy model
class TextureEntropyBottleneck_v3(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck_v3, self).__init__()
        self._ch = int(channels)
        self.s_ch = opt.label_nc + 1
        self.opt = opt

        self.factorized_0 = FullFactorizedModel(self._ch // 4, (3, 3, 3), 1e-9, True)
        if self.s_ch > 32:
            self.cross_region_encoder_0 = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.cross_region_encoder = nn.Sequential(
                nn.Conv2d(self.s_ch, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 16, self.s_ch // 32, 1, 1)
            )
            self.cross_region_decoder = nn.Sequential(
                nn.Conv2d(self.s_ch // 32, self.s_ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 16, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.cross_region_decoder_0 = nn.Sequential(
                nn.Conv2d(self._ch // 4, self._ch, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.cross_region_encoder_0 = nn.Sequential(
                nn.Conv2d(self._ch, self._ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.cross_region_encoder = nn.Sequential(
                nn.Conv2d(self.s_ch, self.s_ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 2, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 8, 1, 1)
            )
            self.cross_region_decoder = nn.Sequential(
                nn.Conv2d(self.s_ch // 8, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 2, self.s_ch, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.cross_region_decoder_0 = nn.Sequential(
                nn.Conv2d(self._ch // 4, self._ch, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
        self.cross_region_parameters = nn.Conv2d(self._ch, self._ch * 2, 1, 1)

        # self.cross_channel_encoder = nn.Sequential(
        #     nn.Conv2d(self._ch, self._ch // 4, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
        # )
        #
        # self.cross_channel_decoder = nn.Sequential(
        #     nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
        # )
        #
        # self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        # self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        # self.hyper_parameter = nn.Sequential(
        #     nn.Conv2d(self._ch * 4, self._ch * 10 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 10 // 3, self._ch * 8 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 8 // 3, self._ch * 2, 1)
        # )

        self.quantizer = Quantizer()
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
        z_0 = self.cross_region_encoder_0(y)
        z_0 = self.cross_region_encoder(torch.transpose(z_0, 1, 2))
        z_0 = torch.transpose(z_0, 1, 2)
        z0_hat, z0_prob = self.factorized_0(z_0)
        u_0 = self.cross_region_decoder(torch.transpose(z0_hat, 1, 2))
        u_0 = self.cross_region_decoder_0(torch.transpose(u_0, 1, 2))
        u_0 = self.cross_region_parameters(u_0)

        # z = self.cross_channel_encoder(y)  # 求先验 [bs,16,19,1]
        # z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        # u = self.cross_channel_decoder(z_hat)  # 先验信息解码

        y_hat = self.quantizer(y, self.training)  # 量化
        # v = self.context_model(y_hat)  # 上下文模型卷积
        # u = self.hyper_parameter(torch.cat((u_0, u), dim=1))  # 上下文与先验融合求概率分布参数

        loc, scale = u_0.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob, y_decor = self.conditional(y_hat, loc, scale)  # 高斯模型求概率

        len_z0 = torch.sum(-torch.log2(z0_prob))
        # len_z = torch.sum(-torch.log2(z_prob))
        len_y = torch.sum(-torch.log2(y_prob))

        length = len_z0 + len_y  # 求出总熵

        return y_hat, z0_hat, length, y_prob


class TextureEntropyBottleneck_v4(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck_v4, self).__init__()
        self._ch = int(channels)
        self.s_ch = opt.label_nc + 1
        self.opt = opt

        self.factorized_0 = FullFactorizedModel(self._ch, (3, 3, 3), 1e-9, True)
        if self.s_ch > 32:
#             self.cross_region_encoder_0 = nn.Sequential(
#                 nn.Conv2d(self._ch, self._ch // 4, 1, 1),
#                 nn.LeakyReLU(inplace=True),
#             )
            self.cross_region_encoder = nn.Sequential(
                nn.Conv2d(self.s_ch, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 16, self.s_ch // 32, 1, 1)
            )
            self.cross_region_decoder = nn.Sequential(
                nn.Conv2d(self.s_ch // 32, self.s_ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 16, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
#             self.cross_region_decoder_0 = nn.Sequential(
#                 nn.Conv2d(self._ch // 4, self._ch, 1, 1),
#                 nn.LeakyReLU(inplace=True),
#             )
        else:
#             self.cross_region_encoder_0 = nn.Sequential(
#                 nn.Conv2d(self._ch, self._ch // 4, 1, 1),
#                 nn.LeakyReLU(inplace=True),
#             )
            self.cross_region_encoder = nn.Sequential(
                nn.Conv2d(self.s_ch, self.s_ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 2, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 8, 1, 1)
            )
            self.cross_region_decoder = nn.Sequential(
                nn.Conv2d(self.s_ch // 8, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 2, self.s_ch, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
#             self.cross_region_decoder_0 = nn.Sequential(
#                 nn.Conv2d(self._ch // 4, self._ch, 1, 1),
#                 nn.LeakyReLU(inplace=True),
#             )
        self.cross_region_parameters = nn.Conv2d(self._ch, self._ch * 2, 1, 1)

        # self.cross_channel_encoder = nn.Sequential(
        #     nn.Conv2d(self._ch, self._ch // 4, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
        # )
        #
        # self.cross_channel_decoder = nn.Sequential(
        #     nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
        # )
        #
        # self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        # self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        # self.hyper_parameter = nn.Sequential(
        #     nn.Conv2d(self._ch * 4, self._ch * 10 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 10 // 3, self._ch * 8 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 8 // 3, self._ch * 2, 1)
        # )

        self.quantizer = Quantizer()
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
#         z_0 = self.cross_region_encoder_0(y)
        z_0 = self.cross_region_encoder(torch.transpose(y, 1, 2))
        z_0 = torch.transpose(z_0, 1, 2)
        z0_hat, z0_prob = self.factorized_0(z_0)
        u_0 = self.cross_region_decoder(torch.transpose(z0_hat, 1, 2))
#         u_0 = self.cross_region_decoder_0(torch.transpose(u_0, 1, 2))
        u_0 = self.cross_region_parameters(torch.transpose(u_0, 1, 2))

        # z = self.cross_channel_encoder(y)  # 求先验 [bs,16,19,1]
        # z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        # u = self.cross_channel_decoder(z_hat)  # 先验信息解码

        y_hat = self.quantizer(y, self.training)  # 量化
        # v = self.context_model(y_hat)  # 上下文模型卷积
        # u = self.hyper_parameter(torch.cat((u_0, u), dim=1))  # 上下文与先验融合求概率分布参数

        loc, scale = u_0.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob, y_decor = self.conditional(y_hat, loc, scale)  # 高斯模型求概率

        len_z0 = torch.sum(-torch.log2(z0_prob))
        # len_z = torch.sum(-torch.log2(z_prob))
        len_y = torch.sum(-torch.log2(y_prob))

        length = len_z0 + len_y  # 求出总熵

        return y_hat, z0_hat, length, y_prob


# enlarge convolution kernel
class TextureEntropyBottleneck_v5(BaseNetwork):
    def __init__(self, channels, opt):
        super(TextureEntropyBottleneck_v5, self).__init__()
        self._ch = int(channels)
        self.s_ch = opt.label_nc + 1
        self.opt = opt

        self.factorized_0 = FullFactorizedModel(self._ch, (3, 3, 3), 1e-9, True)
        if self.s_ch > 32:
            # B,N,
            self.cross_region_encoder = nn.Sequential(
                nn.Conv2d(self.s_ch, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 16, self.s_ch // 32, 1, 1)
            )
            self.cross_region_decoder = nn.Sequential(
                nn.Conv2d(self.s_ch // 32, self.s_ch // 16, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 16, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
#             self.cross_region_decoder_0 = nn.Sequential(
#                 nn.Conv2d(self._ch // 4, self._ch, 1, 1),
#                 nn.LeakyReLU(inplace=True),
#             )
        else:
#             self.cross_region_encoder_0 = nn.Sequential(
#                 nn.Conv2d(self._ch, self._ch // 4, 1, 1),
#                 nn.LeakyReLU(inplace=True),
#             )
            self.cross_region_encoder = nn.Sequential(
                nn.Conv2d(self.s_ch, self.s_ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 2, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 8, 1, 1)
            )
            self.cross_region_decoder = nn.Sequential(
                nn.Conv2d(self.s_ch // 8, self.s_ch // 4, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 4, self.s_ch // 2, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.s_ch // 2, self.s_ch, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
#             self.cross_region_decoder_0 = nn.Sequential(
#                 nn.Conv2d(self._ch // 4, self._ch, 1, 1),
#                 nn.LeakyReLU(inplace=True),
#             )
        self.cross_region_parameters = nn.Conv2d(self._ch, self._ch * 2, 1, 1)

        # self.cross_channel_encoder = nn.Sequential(
        #     nn.Conv2d(self._ch, self._ch // 4, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 4, self._ch // 16, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 16, self._ch // 32, 1, 1)
        # )
        #
        # self.cross_channel_decoder = nn.Sequential(
        #     nn.Conv2d(self._ch // 32, self._ch // 16, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 16, self._ch // 2, 1, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch // 2, self._ch * 2, 1, 1)
        # )
        #
        # self.factorized = FullFactorizedModel(self._ch // 32, (3, 3, 3), 1e-9, True)

        # self.context_model = MaskedConv2d('A', self._ch, self._ch * 2, 5, padding=2)

        # self.hyper_parameter = nn.Sequential(
        #     nn.Conv2d(self._ch * 4, self._ch * 10 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 10 // 3, self._ch * 8 // 3, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(self._ch * 8 // 3, self._ch * 2, 1)
        # )

        self.quantizer = Quantizer()
        self.conditional = GaussianModel(1e-1, 1e-9, True)

    def forward(self, y):
#         z_0 = self.cross_region_encoder_0(y)
        z_0 = self.cross_region_encoder(torch.transpose(y, 1, 2))
        z_0 = torch.transpose(z_0, 1, 2)
        z0_hat, z0_prob = self.factorized_0(z_0)
        u_0 = self.cross_region_decoder(torch.transpose(z0_hat, 1, 2))
#         u_0 = self.cross_region_decoder_0(torch.transpose(u_0, 1, 2))
        u_0 = self.cross_region_parameters(torch.transpose(u_0, 1, 2))

        # z = self.cross_channel_encoder(y)  # 求先验 [bs,16,19,1]
        # z_hat, z_prob = self.factorized(z)  # 先验概率估计 # independent prob, z_hat: quantized and noised z
        # u = self.cross_channel_decoder(z_hat)  # 先验信息解码

        y_hat = self.quantizer(y, self.training)  # 量化
        # v = self.context_model(y_hat)  # 上下文模型卷积
        # u = self.hyper_parameter(torch.cat((u_0, u), dim=1))  # 上下文与先验融合求概率分布参数

        loc, scale = u_0.split(self._ch, dim=1)  # 参数分成 方差，均值
        y_prob, y_decor = self.conditional(y_hat, loc, scale)  # 高斯模型求概率

        len_z0 = torch.sum(-torch.log2(z0_prob))
        # len_z = torch.sum(-torch.log2(z_prob))
        len_y = torch.sum(-torch.log2(y_prob))

        length = len_z0 + len_y  # 求出总熵

        return y_hat, z0_hat, length, y_prob
