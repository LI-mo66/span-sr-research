"""Strict mapping for read-only timing diagnostics, never used by run.py."""
def load_frozen_reference(model,state):
    mapped=model.state_dict();seen=set()
    for name,value in state.items():
        key=name.replace('block_3.spatial_gate.depth.','block_3.spatial_gate.depth.local.') if name.startswith('block_3.spatial_gate.depth.') else name
        if key not in mapped or mapped[key].shape!=value.shape:raise ValueError('Unexpected reference key/shape: '+name)
        mapped[key]=value;seen.add(key)
    prefix='block_3.spatial_gate.depth.'
    extra={prefix+x for x in ('horizontal.weight','horizontal.bias','vertical.weight','vertical.bias')}
    if set(mapped)!=seen|extra:raise ValueError('Reference coverage mismatch')
    mapped[prefix+'vertical.weight'].zero_();mapped[prefix+'vertical.bias'].zero_()
    model.load_state_dict(mapped,strict=True)
    return dict(original_keys=len(state),mapped_original_keys=len(seen),new_keys=len(extra),missing=[],unexpected=[])
