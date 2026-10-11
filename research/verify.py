"""Engineering checks only; synthetic images are not accuracy evidence."""
import copy
import json
import tempfile
from pathlib import Path
import torch
from runtime import components

def main():
    torch.set_num_threads(2)
    arch, _ = components()
    torch.manual_seed(10)
    model = arch.SPAN(3, 3, upscale=4, feature_channels=48, img_range=255.)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=5e-4)
    before = model.conv_1.conv[0].weight.detach().clone()
    x = torch.rand(1, 3, 13, 17)
    loss = (model(x)-torch.rand(1,3,52,68)).abs().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert not torch.equal(before, model.conv_1.conv[0].weight)
    model.eval()
    with torch.no_grad(): y = model(x)
    clone = copy.deepcopy(model)
    tmp_root = Path(__file__).resolve().parents[1] / 'tmp'
    tmp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        path = Path(tmp)/'checkpoint.pt'
        torch.save(model.state_dict(), path)
        clone.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad(): assert torch.equal(y, clone(x))
    # Structural reparameterization must agree before nonlinear SPAB paths.
    c = arch.Conv3XC(3, 5, gain1=2)
    with torch.no_grad():
        train_out = c(x)
        c.eval(); eval_out = c(x)
    torch.testing.assert_close(train_out, eval_out, atol=1e-6, rtol=1e-5)
    print(json.dumps(dict(status='PASS', device='CPU', shape=list(y.shape), loss=float(loss.detach()), total_params=sum(p.numel() for p in model.parameters()), trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad), evidence='synthetic engineering verification only'), indent=2))

if __name__ == '__main__': main()
