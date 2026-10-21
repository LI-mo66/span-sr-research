from typing import Tuple, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY

class CSA(nn.Module):
    def __init__(self, dim):
        '''
        Cross-Symmetric Aggregation Module
        '''
        super(CSA, self).__init__()

        self.hidden_dim = dim // 2

        self.conv_0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(dim, self.hidden_dim, 3, padding=3 // 2, groups=self.hidden_dim)
        self.conv_2 = nn.Conv2d(dim, self.hidden_dim, 3, padding=3 // 2, groups=self.hidden_dim)
        self.act = nn.GELU()
        self.conv_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=3 // 2, groups=self.hidden_dim)
        self.conv_4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=3 // 2, groups=self.hidden_dim)
        # self.proj_out = nn.Conv2d(self.hidden_dim, dim, 1, 1, 0)
        self.proj_out = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x, k):
        x = self.act(self.conv_0(x))
        x1 = self.conv_1(x)
        x2 = self.conv_2(k)
        x = torch.cat([self.conv_3(x1 * self.act(x2)), self.conv_4(x2 * self.act(x1))], dim=1)

        return self.proj_out(x)

class Router(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class StripedConv2d(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int,
                 depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, self.kernel_size), padding=(0, self.padding),
                      groups=in_ch if depthwise else 1),
            nn.Conv2d(in_ch, in_ch, kernel_size=(self.kernel_size, 1), padding=(self.padding, 0),
                      groups=in_ch if depthwise else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class MSP(nn.Module):
    def __init__(self, in_ch, recursive):
        super().__init__()
        self.conv2 = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.conv3 = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.conv4 = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.conv5 = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.DWConv_32 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.DWConv_33 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.DWConv_34 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.DWConv_35 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.act = nn.GELU()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.recursive = recursive

    def forward(self, x):
        res_1 = self.sigmoid(self.AdaptiveAvgPool(x))
        res_2 = F.adaptive_max_pool2d(x, (x.size(2) // 2, x.size(3) // 2))
        res_3 = F.adaptive_max_pool2d(x, (x.size(2) // 4, x.size(3) // 4))
        res_4 = F.adaptive_max_pool2d(x, (x.size(2) // 8, x.size(3) // 8))
        res_5 = F.adaptive_max_pool2d(x, (x.size(2) // 16, x.size(3) // 16))

        for _ in range(self.recursive):
            res_2 = self.act(self.conv2(self.DWConv_32(res_2)))
            res_3 = self.act(self.conv3(self.DWConv_33(res_3)))
            res_4 = self.act(self.conv4(self.DWConv_34(res_4)))
            res_5 = self.act(self.conv5(self.DWConv_35(res_5)))

        res_2 = F.interpolate(res_2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        res_3 = F.interpolate(res_3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        res_4 = F.interpolate(res_4, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        res_5 = F.interpolate(res_5, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        mutil_out = res_2 + res_3 + res_4 + res_5
        x_res = res_1 * mutil_out
        return x_res

class HFAB(nn.Module):
    '''
    Hybrid Feature Aggregation Block (HFAB)
    '''

    def __init__(self, channels, recursive):
        super(HFAB, self).__init__()
        self.msp = MSP(channels, recursive)
        self.DWConv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.DWConv3(x)
        x2 = self.msp(x)
        out = self.act(self.conv1(torch.cat((x1, x2), dim=1)))
        return out

class Attention(nn.Module):

    def __init__(self, embed_dim, fft_norm="ortho"):
        super(Attention, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_layer2 = torch.nn.Conv2d(embed_dim // 2, embed_dim // 2, 1, 1, 0)
        self.conv_layer3 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real = ffted.real + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.real))))
        )
        imag = ffted.imag + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.imag))))
        )

        ffted = torch.complex(real, imag)
        ifft_shape_slice = x.shape[-2:]
        atten = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )

        return x * atten

class HFDB(nn.Module):
    '''
    Hybrid Feature Distillation Block (HFDB)
    '''

    def __init__(self, channels, recursive):
        super(HFDB, self).__init__()
        self.c1_d = nn.Conv2d(channels, channels // 2, 1)
        self.c1_r = HFAB(channels, recursive)
        self.c2_d = nn.Conv2d(channels, channels // 2, 1)
        self.c2_r = HFAB(channels, recursive)
        self.c3_d = nn.Conv2d(channels, channels // 2, 1)
        self.c3_r = HFAB(channels, recursive)
        self.c4 = nn.Conv2d(channels, channels // 2, 1)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(channels * 2, channels, 1)
        self.attn = Attention(channels)
        self.c6 = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.act(self.c5(out))
        out = self.attn(out)
        out = self.c6(out)
        return out

#################
# MoE Layer
#################
class MoEBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 use_shuffle: bool = False,
                 lr_space: str = "linear",
                 recursive: int = 2):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive

        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0)
        # )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, 2 * in_ch, 1, 1, 0),
            nn.GELU()
        )

        self.hfdb = HFDB(in_ch, self.recursive)

        # 改为池化分支
        # self.agg_conv = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
        #     # nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=3 // 2, groups=in_ch),
        #     nn.GELU())

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        )

        self.conv_2 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True),
            nn.GELU())

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer(
            experts=[CSA(dim=in_ch) for i in range(num_experts)],
            # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hfdb(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)

        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x)
        k = self.calibrate(k)

        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        out = inputs.clone()

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i:i + 1, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i:i + 1, None, None]

        return out

def rotate_every_two(x):
    """
    实现RoPE（旋转位置编码）中的成对旋转操作
    将输入张量按通道维度两两分组，对每组进行旋转（x1→-x2, x2→x1）
    Args:
        x: 输入张量，形状为(b, n_heads, seq_len, head_dim)
    Returns:
        旋转后的张量，形状与输入保持一致
    """
    x1 = x[:, :, :, ::2]  # 取偶索引通道
    x2 = x[:, :, :, 1::2]  # 取奇索引通道
    x = torch.stack([-x2, x1], dim=-1)  # 堆叠旋转后的通道
    return x.flatten(-2)  # 展平最后两个维度，恢复原形状


def theta_shift(x, sin, cos):
    """
    应用RoPE旋转位置编码
    通过sin和cos将输入张量进行位置偏移，注入位置信息
    Args:
        x: 输入张量（query或key）
        sin: 位置编码的正弦分量
        cos: 位置编码的余弦分量
    Returns:
        注入位置信息后的张量
    """
    return (x * cos) + (rotate_every_two(x) * sin)


class RoPE(nn.Module):
    """
    二维旋转位置编码（Rotary Position Embedding）模块
    为2D特征（如图像）生成位置相关的sin/cos编码，支持高度和宽度维度的位置信息注入
    """

    def __init__(self, embed_dim, num_heads=4):
        """
        Args:
            embed_dim: 总嵌入维度
            num_heads: 注意力头数量
        """
        super().__init__()
        # 生成角度参数（10000^(-2i/d)），用于计算位置编码
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()  # 扩展为成对角度
        self.register_buffer('angle', angle)  # 注册为非训练参数

    def forward(self, slen: Tuple[int]):
        """
        生成二维位置编码的sin和cos分量
        Args:
            slen: 特征的空间尺寸(h, w)
        Returns:
            展平后的sin和cos编码，形状为((h*w), head_dim)
        """
        h, w = slen
        # 生成高度和宽度维度的索引
        index_h = torch.arange(h).to(self.angle)
        index_w = torch.arange(w).to(self.angle)

        # 计算高度相关的sin/cos编码并扩展到整个空间
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h, d/2)
        sin_h = sin_h.unsqueeze(1).repeat(1, w, 1)  # (h, w, d/2)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h, d/2)
        cos_h = cos_h.unsqueeze(1).repeat(1, w, 1)  # (h, w, d/2)

        # 计算宽度相关的sin/cos编码并扩展到整个空间
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w, d/2)
        sin_w = sin_w.unsqueeze(0).repeat(h, 1, 1)  # (h, w, d/2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w, d/2)
        cos_w = cos_w.unsqueeze(0).repeat(h, 1, 1)  # (h, w, d/2)

        # 拼接高度和宽度编码，展平为序列形式
        sin = torch.cat([sin_h, sin_w], -1).flatten(0, 1)  # (h*w, d)
        cos = torch.cat([cos_h, cos_w], -1).flatten(0, 1)  # (h*w, d)
        return (sin, cos)


class GateLinearAttentionNoSilu(nn.Module):

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        '''
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        '''
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.dim, :, :]
        o = qkvo[:, 3*self.dim:, :, :]
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :]) # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads) # (b n (h w) d)

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0 # (b n l d)

        q_mean = q.mean(dim=-2, keepdim=True) # (b n 1 d)
        eff = self.scale * q_mean @ k.transpose(-1, -2) # (b n 1 l)
        eff = torch.softmax(eff, dim=-1).transpose(-1, -2) # (b n l 1)
        k = k * eff * (H*W)

        q_rope = theta_shift(q, sin, cos)
        k_rope = theta_shift(k, sin, cos)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6) # (b n l 1)
        kv = (k_rope.transpose(-2, -1) * ((H*W) ** -0.5)) @ (v * ((H*W) ** -0.5)) # (b n d d)

        res = q_rope @ kv * z # (b n l d)
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)


class PDFN(nn.Module):
    def __init__(self, dim, expansion_factor):
        super(PDFN, self).__init__()
        h_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, h_dim, kernel_size=1)

        self.dwconv = nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, groups=h_dim)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1),
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = x * self.sca(x)
        x = self.project_out(x)
        return x

#############################
# Components
#############################
class ResGroup(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 global_kernel_size: int = 11,
                 lr_space: int = 1,
                 topk: int = 2,
                 recursive: int = 2,
                 use_shuffle: bool = False):
        super().__init__()

        self.local_block = RME(in_ch=in_ch,
                               num_experts=num_experts,
                               use_shuffle=use_shuffle,
                               lr_space=lr_space,
                               topk=topk,
                               recursive=recursive)
        self.global_block = SME(in_ch=in_ch,
                                kernel_size=global_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_block(x)
        x = self.global_block(x)
        return x


#############################
# Global Block
#############################
class SME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int = 11):
        super().__init__()

        self.rope = RoPE(in_ch)

        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = GateLinearAttentionNoSilu(in_ch)

        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = PDFN(in_ch, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        sin, cos = self.rope((h, w))
        x = self.block(self.norm_1(x), sin, cos) + x
        x = self.ffn(self.norm_2(x)) + x
        return x

#############################
# Local Blocks
#############################
class RME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 lr_space: int = 1,
                 recursive: int = 2,
                 use_shuffle: bool = False, ):
        super().__init__()

        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = MoEBlock(in_ch=in_ch, num_experts=num_experts, topk=topk, use_shuffle=use_shuffle,
                              recursive=recursive, lr_space=lr_space)

        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = PDFN(in_ch, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x

def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


######################
# Meta Architecture
######################
# @ARCH_REGISTRY.register()
class SeemoRe16(nn.Module):
    def __init__(self,
                 scale: int = 4,
                 in_chans: int = 3,
                 num_experts: int = 6,
                 num_layers: int = 6,
                 embedding_dim: int = 64,
                 img_range: float = 1.0,
                 use_shuffle: bool = False,
                 global_kernel_size: int = 11,
                 recursive: int = 2,
                 lr_space: int = 1,
                 topk: int = 2,):
        super().__init__()
        self.scale = scale
        self.num_in_channels = in_chans
        self.num_out_channels = in_chans
        self.img_range = img_range
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        
        
        # -- SHALLOW FEATURES --
        self.conv_1 = nn.Conv2d(self.num_in_channels, embedding_dim, kernel_size=3, padding=1)
        
        # -- DEEP FEATURES --
        self.body = nn.ModuleList(
            [ResGroup(in_ch=embedding_dim, 
                       num_experts=num_experts, 
                       use_shuffle=use_shuffle,
                       topk=topk,
                       lr_space=lr_space,
                       recursive=recursive,
                       global_kernel_size=global_kernel_size) for i in range(num_layers)]
        )
        
        # -- UPSCALE --
        self.norm = LayerNorm(embedding_dim, data_format='channels_first')
        self.conv_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embedding_dim, (scale**2) * self.num_out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        # -- SHALLOW FEATURES --
        x = self.conv_1(x)
        res = x
        
        # -- DEEP FEATURES --
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.norm(x)
                
        # -- HR IMAGE RECONSTRUCTION --
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x


if __name__ == '__main__':

    # TODO thop计算方式
    from thop import profile
    from thop import clever_format

    upscale = 4
    height = (1280 // upscale)
    width = (720 // upscale)

    '''
    network_g:
    type: SeemoRe
    scale: 4
    in_chans: 3
    num_experts: 3
    img_range: 1.0
    num_layers: 6
    embedding_dim: 36
    use_shuffle: True
    lr_space: exp
    topk: 1
    recursive: 2
    global_kernel_size: 11
    '''

    model = SeemoRe16(scale=upscale, in_chans=3, num_experts=1, img_range=1.0, num_layers=4, embedding_dim=32, use_shuffle=True, lr_space='exp', topk=1, recursive=2, global_kernel_size=11)
    input = torch.randn(1, 3, height, width)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = input.to(device)
    macs, params = profile(model.to(device), inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs:", macs)
    print("params:", params)