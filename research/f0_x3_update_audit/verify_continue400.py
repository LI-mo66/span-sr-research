"""CPU-only manager quality fixtures and actual MultiStep boundary check."""
import copy
from pathlib import Path
import torch
from continue_to400 import expected_lr, quality_stop

source=Path(__file__).with_name('verify_quality_gate.py').read_text(encoding='utf-8')
source=source.replace('from continue_to300 import quality_stop','')
exec(compile(source,'quality_fixtures','exec'))
assert list(range(311000,400001,1000))[-1]==400000
assert len(list(range(311000,400001,1000)))==90
assert expected_lr(399999)==1e-4
assert expected_lr(400000)==5e-5
param=torch.nn.Parameter(torch.ones(()))
opt=torch.optim.Adam([param],lr=2e-4)
scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[200000,400000,600000,800000],gamma=.5)
state=scheduler.state_dict()
state.update(last_epoch=399999,_last_lr=[1e-4],_step_count=400000)
scheduler.load_state_dict(state)
opt.param_groups[0]['lr']=1e-4
for update,used,after in ((400000,1e-4,5e-5),(400001,5e-5,5e-5)):
    assert opt.param_groups[0]['lr']==used
    param.grad=torch.ones_like(param)
    opt.step();scheduler.step()
    assert scheduler.last_epoch==update
    assert scheduler.get_last_lr()==[after]
    assert opt.param_groups[0]['lr']==expected_lr(update)
print('PASS: 90 stages, existing quality gates, actual Adam/MultiStep 400K boundary')
