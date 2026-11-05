"""Exact LKSA class from locked Apache-2.0 UCAN source; see vendor/manifest.json."""
import torch
from torch import nn

class LKSA(nn.Module):
    r"""
    Large-Kernel Spatial Attention enhanced with an attention-based (Global Context)
    dynamic 3x3 core. This version is self-contained and does not use inheritance.

    Args
    ----
    dim  : input / output channels
    k    : total kernel size of the depth-wise spatial filter
           (must be odd and one of {7, 11, 23, 35, 41, 53})
    ratio: bottleneck ratio for the dynamic-kernel MLP
    """
    STATIC_SPECS = {
        7 : dict(core=3, dilate=1),
        11: dict(core=5, dilate=1),
        23: dict(core=5, dilate=2),
        35: dict(core=5, dilate=3, extra=11),
        41: dict(core=5, dilate=3, extra=13),
        53: dict(core=5, dilate=3, extra=17),
    }
    def __init__(self, dim: int, k: int = 35, ratio: int = 4):
        super().__init__()
        assert k in self.STATIC_SPECS, f"Unsupported k={k}"
        spec = self.STATIC_SPECS[k]
        core_k    = spec["core"]
        dil       = spec["dilate"]
        extra_k   = spec.get("extra", 0)

        # ---------- static LKA rim (depth-wise) ----------
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, core_k), padding=(0, (core_k-1)//2), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=(core_k, 1), padding=((core_k-1)//2, 0), groups=dim)
        self.conv_dh = nn.Conv2d(dim, dim, kernel_size=(1, core_k), padding=(0, dil * (core_k//2)), dilation=dil, groups=dim)
        self.conv_dv = nn.Conv2d(dim, dim, kernel_size=(core_k, 1), padding=(dil * (core_k//2), 0), dilation=dil, groups=dim)

        if extra_k:  # k >= 35
            self.conv_big_h = nn.Conv2d(dim, dim, kernel_size=(1, extra_k), padding=(0, dil * (extra_k//2)), dilation=dil, groups=dim)
            self.conv_big_v = nn.Conv2d(dim, dim, kernel_size=(extra_k, 1), padding=(dil * (extra_k//2), 0), dilation=dil, groups=dim)
        else:
            self.conv_big_h = self.conv_big_v = nn.Identity()

        # ----------  3x3 ----------
        mid = int(dim * ratio)
        self.conv3 = nn.Sequential(nn.Conv2d(dim, dim // ratio, 1, 1, 0),
                        nn.GELU(),
                        nn.Conv2d(dim // ratio, dim // ratio, 3, 1, 1),
                        nn.GELU(),
                        nn.Conv2d(dim // ratio, dim, 1, 1, 0))

        self.k_size = k  # full spatial size (after padding)

        # 1x1 fuse
        self.channel = nn.Conv2d(dim, dim, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LKSA.
        """
        B, C, H, W = x.shape

        # ---------------- static LKA pathway ----------------
        y = self.conv_h(x)
        y = self.conv_v(y)
        y = self.conv_dh(y)
        y = self.conv_dv(y)
        y = self.conv_big_h(y)
        large_kn = self.conv_big_v(y)

        out = self.channel(x) * (large_kn + self.conv3(x))
       
        return out
