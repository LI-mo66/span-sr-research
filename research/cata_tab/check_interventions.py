"""Check SDPA interception against independent value-projection bypasses."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tab import ContentTAB
from diagnose_dataset import intervention


def main():
    torch.set_num_threads(2)
    torch.manual_seed(10)
    for shape in ((1, 52, 1, 1), (1, 52, 3, 43)):
        tab = ContentTAB(chunk_groups=1)
        image = torch.randn(*shape)
        with torch.no_grad():
            tab(image)  # Initialize centers, then freeze.
        tab.eval()
        model = SimpleNamespace(content_tab=tab)
        frozen = {k: v.clone() for k, v in tab.state_dict().items()}
        with torch.inference_mode():
            for name in ('group_disabled', 'center_disabled', 'both_disabled'):
                with intervention(model, name, shape[2]*shape[3]):
                    actual = tab(image)
                modules = []
                if name in ('group_disabled', 'both_disabled'):
                    modules.append(tab.iasa_attn.to_v)
                if name in ('center_disabled', 'both_disabled'):
                    modules.append(tab.irca_attn.to_v)
                handles = [module.register_forward_hook(lambda m, i, o: torch.zeros_like(o)) for module in modules]
                try:
                    expected = tab(image)
                finally:
                    for h in handles:
                        h.remove()
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            with intervention(model, 'tab_identity', shape[2]*shape[3]):
                assert torch.equal(tab(image), image)
            with intervention(model, 'both_disabled', shape[2]*shape[3]):
                residual = image.flatten(2).transpose(1, 2)
                expected = (residual + tab.mlp(residual, shape[2:])).transpose(1, 2).reshape_as(image)
                torch.testing.assert_close(tab(image), expected, atol=1e-6, rtol=1e-6)
            with patch.object(tab.mlp, 'forward', lambda x, size: torch.zeros_like(x)):
                expected = tab(image)
            with intervention(model, 'ffn_disabled', shape[2]*shape[3]):
                assert torch.equal(tab(image), expected)
        assert all(torch.equal(v, frozen[k]) for k, v in tab.state_dict().items())
    print('PASS tiny/multiple-chunk routing, independent value-zero controls, residuals, frozen state')


if __name__ == '__main__':
    main()
