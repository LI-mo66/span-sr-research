"""Trace real loss through both TAB paths and test full SPAN state lifecycle."""
import argparse
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cata_runtime', HERE / 'runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
import torch
from torch import nn
from model import build_model
from deploy import fused_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arch, _ = runtime.components()
    cfg = json.loads((HERE / 'configs/C1-mid-lr2e4.json').read_text())
    torch.manual_seed(cfg['seed'])
    baseline = build_model(arch, dict(cfg, cata_tab=False)).to(args.device)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    torch.manual_seed(cfg['seed'])
    model = build_model(arch, cfg).to(args.device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items())
    shapes = ((1,3,1,1), (2,3,3,5), (1,3,3,43), (2,3,13,17), (1,3,64,64), (1,3,65,71))
    for shape in shapes:
        value = torch.rand(*shape, device=args.device)
        before = value.clone()
        for training in (True, False):
            model.train(training)
            with torch.no_grad():
                output = model(value)
                assert output.shape == (shape[0],3,shape[2]*4,shape[3]*4)
                assert output.dtype == value.dtype and torch.isfinite(output).all()
        assert torch.equal(value,before)
    # Check wiring alone with Identity: no double addition or dropped experts.
    adapter = model.content_tab
    model.content_tab = nn.Identity()
    model.eval()
    baseline.eval()
    value = torch.rand(2,3,13,17,device=args.device)
    with torch.no_grad():
        torch.testing.assert_close(model(value),baseline(value),rtol=0,atol=0)
    model.content_tab = adapter
    initial_means = adapter.means.clone()
    initial_flag = adapter.initted.clone()
    captured_features = []
    capture = adapter.register_forward_pre_hook(
        lambda _, inputs: captured_features.append(inputs[0].detach().clone()))
    with torch.no_grad():
        together = model(value)
    capture.remove()
    with torch.no_grad():
        separate = torch.cat([model(value[i:i+1]) for i in range(2)])
        # Check image isolation on identical feature tensors, independently of
        # shape-dependent upstream convolution rounding and hard routing.
        features = captured_features[0]
        adapter_batch_rounding_max = float((adapter(features) -
            torch.cat([adapter(features[i:i+1]) for i in range(2)])).abs().max())
        # FP64 isolates mathematical sample independence from FP32 GEMM/conv
        # backend rounding. Hard routing is numerically sensitive near ties.
        precise = copy.deepcopy(adapter).double()
        precise_features = features.double()
        torch.testing.assert_close(precise(precise_features),
            torch.cat([precise(precise_features[i:i+1]) for i in range(2)]),
            rtol=1e-10,atol=1e-11)
        batch_output_rounding_max = float((together-separate).abs().max())
        baseline_batch_difference = float((baseline(value) - torch.cat(
            [baseline(value[i:i+1]) for i in range(2)])).abs().max())
    assert torch.equal(initial_means,adapter.means) and torch.equal(initial_flag,adapter.initted)
    # Real final-output loss, not a surrogate loss on a hidden feature.
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable,lr=cfg['lr'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'])
    optimizer_ids = [id(p) for group in opt.param_groups for p in group['params']]
    assert len(optimizer_ids) == len(set(optimizer_ids)) == len(trainable)
    before = {k:p.detach().clone() for k,p in adapter.named_parameters()}
    gradients, losses, branch_gradients = {}, [], {'group':[], 'center':[]}
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = [0]
    def observed_attention(*inputs, **kwargs):
        result = original_sdpa(*inputs, **kwargs)
        branch = 'group' if calls[0] % 2 == 0 else 'center'
        calls[0] += 1
        if result.requires_grad:
            result.register_hook(lambda grad: branch_gradients[branch].append(float(grad.abs().max())))
        return result
    target = torch.rand(2,3,52,68,device=args.device)
    saved = None
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        with patch('torch.nn.functional.scaled_dot_product_attention', observed_attention):
            loss = torch.nn.functional.l1_loss(model(value), target)
            assert torch.isfinite(loss)
            loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        if step == 0:
            gradients = {k:float(p.grad.abs().max()) for k,p in adapter.named_parameters()}
            assert all(v>0 for v in gradients.values())
        opt.step()
        losses.append(float(loss.detach()))
        if step == 1:
            # Serialize the real optimizer and module buffers at the same point.
            saved = io.BytesIO()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),saved)
    assert losses[-1] < losses[0], losses
    updated = {k:not torch.equal(before[k],p) for k,p in adapter.named_parameters()}
    assert all(updated.values())
    assert all(values and min(values)>0 for values in branch_gradients.values())
    assert not torch.equal(initial_means,adapter.means)
    # Replay steps 3/4 from the serialized second step (no stochastic layers).
    saved.seek(0)
    state = torch.load(saved,map_location=args.device,weights_only=True)
    replay = build_model(arch,cfg).to(args.device).train()
    replay.load_state_dict(state['model'],strict=True)
    replay_opt = torch.optim.Adam([p for p in replay.parameters() if p.requires_grad],lr=cfg['lr'])
    replay_opt.load_state_dict(state['optimizer'])
    for _ in range(2):
        replay_opt.zero_grad(set_to_none=True)
        torch.nn.functional.l1_loss(replay(value),target).backward()
        replay_opt.step()
    assert all(torch.equal(v,replay.state_dict()[k]) for k,v in model.state_dict().items())
    model.eval()
    actual_features = []
    handle = adapter.register_forward_pre_hook(
        lambda _, inputs: actual_features.append(inputs[0].detach().clone()))
    with torch.no_grad():
        prediction = model(value)
    handle.remove()
    with torch.no_grad():
        buffers = {k:v.clone() for k,v in adapter.named_buffers()}
        _, stats = adapter.forward_with_diagnostics(actual_features[0])
        assert all(torch.equal(v,buffers[k]) for k,v in adapter.named_buffers())
        assert stats['attention_delta_ratio']>0 and stats['ffn_delta_ratio']>0
    clone = build_model(arch,cfg).to(args.device).eval()
    clone.load_state_dict(model.state_dict(),strict=True)
    deployed = fused_copy(model,arch.Conv3XC)
    with torch.no_grad():
        torch.testing.assert_close(prediction,clone(value),rtol=0,atol=0)
        torch.testing.assert_close(prediction,deployed(value),rtol=1e-5,atol=1e-6)
    assert model.extra_parameters == 32812
    assert sum(p.numel() for p in deployed.parameters()) == 531020
    report = dict(status='PASS',device=args.device,torch=torch.__version__,
                  shapes=[list(s) for s in shapes],extra_parameters=32812,
                  deployed_parameters=531020,adapter_first_gradient_max=gradients,
                  adapter_parameters_updated=updated,
                  group_attention_output_gradient_min=min(branch_gradients['group']),
                  center_attention_output_gradient_min=min(branch_gradients['center']),
                  synthetic_l1=losses,diagnostic_example=stats,
                  full_network_batch_difference_max=batch_output_rounding_max,
                  baseline_batch_difference_max=baseline_batch_difference,
                  adapter_batch_fp32_rounding_max=adapter_batch_rounding_max,
                  checks=['baseline weights/RNG','original SPAN routes with Identity',
                          'shape/tiny/odd/non-square','eval buffers frozen and batch independence',
                          'both attention paths reach final RGB L1','all new parameter tensors update',
                          'center buffers update in train','optimizer coverage',
                          'serialized optimizer replay exact','strict reload','prefused agreement'],
                  evidence='engineering/mechanism only; no real PSNR measured')
    if args.output:
        Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
