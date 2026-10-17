"""Single content TAB at SPAB3; all original SPAN routes remain intact."""
import torch

from tab import ContentTAB


def build_model(arch, cfg):
    if cfg.get('variants') or cfg.get('moe_fs') or cfg.get('easa'):
        raise ValueError('Content-TAB experiment cannot mix other candidates')
    if (cfg['model']['feature_channels'] != 52 or cfg['model']['upscale'] != 4 or
            cfg['model']['num_in_ch'] != 3 or cfg['model']['num_out_ch'] != 3):
        raise ValueError('Experiment requires RGB SPAN52 x4')
    enabled = cfg.get('cata_tab', False)
    if type(enabled) is not bool:
        raise ValueError('cata_tab must be boolean')
    if not enabled:
        return arch.SPAN(**cfg['model'])
    settings = cfg.get('cata_settings', {})
    if set(settings) - {'chunk_groups'}:
        raise ValueError('Structural settings are fixed for ContentTAB-v1')

    class ContentSPAN(arch.SPAN):
        def __init__(self):
            super().__init__(**cfg['model'])
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg['seed'] + 30001)
                self.content_tab = ContentTAB(chunk_groups=settings.get('chunk_groups', 128))
            self.extra_parameters = sum(p.numel() for p in self.content_tab.parameters())
            if self.extra_parameters > cfg['max_extra_deployed_parameters']:
                raise ValueError('Deployment parameter budget exceeded')

        def forward(self, x):
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            feature = self.conv_1(x)
            b1, _, _ = self.block_1(feature)
            b2, _, _ = self.block_2(b1)
            b3, _, _ = self.block_3(b2)
            b3 = self.content_tab(b3)  # TAB already includes its own two residuals
            b4, _, _ = self.block_4(b3)
            b5, _, _ = self.block_5(b4)
            b6, b6_first, _ = self.block_6(b5)
            b6 = self.conv_2(b6)
            fused = self.conv_cat(torch.cat([feature, b6, b1, b6_first], 1))
            return self.upsampler(fused)

    return ContentSPAN()
