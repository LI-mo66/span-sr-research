"""Residual ATD attention only; original SPAN routes remain intact."""
import torch

from adapter import AttentionAdapter


def build_model(arch, cfg):
    if cfg.get('variants') or cfg.get('moe_fs') or cfg.get('easa') or cfg.get('cata_tab') or cfg.get('cata_ffn'):
        raise ValueError('ATD attention experiment cannot mix other candidates')
    if (cfg['model']['feature_channels'] != 52 or cfg['model']['upscale'] != 4 or
            cfg['model']['num_in_ch'] != 3 or cfg['model']['num_out_ch'] != 3):
        raise ValueError('Experiment requires RGB SPAN52 x4')
    enabled = cfg.get('atd_attention', False)
    if type(enabled) is not bool:
        raise ValueError('atd_attention must be boolean')
    if not enabled:
        return arch.SPAN(**cfg['model'])
    settings = cfg.get('atd_settings', {})
    if set(settings) - {'position','query_chunk','group_chunk'}:
        raise ValueError('Structural settings are fixed for ATD attention-v1')
    position = settings.get('position')
    if position != 'tail':
        raise ValueError('ATD attention control requires tail position')

    class ContentSPAN(arch.SPAN):
        def __init__(self):
            super().__init__(**cfg['model'])
            self.content_position = position
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(cfg['seed'] + 30001)
                self.content_tab = AttentionAdapter(query_chunk=settings.get('query_chunk',65536), group_chunk=settings.get('group_chunk',256))
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
            b4, _, _ = self.block_4(b3)
            b5, _, _ = self.block_5(b4)
            b6, b6_first, _ = self.block_6(b5)
            b6 = self.conv_2(b6)
            fused = self.conv_cat(torch.cat([feature, b6, b1, b6_first], 1))
            fused = self.content_tab(fused)  # Exactly one attention residual.
            return self.upsampler(fused)

    return ContentSPAN()
