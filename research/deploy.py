"""Explicit pre-fusion export; never modifies the retained upstream model."""
import copy

def fused_copy(model, conv_type):
    result=copy.deepcopy(model).eval()
    def visit(parent):
        for name,child in list(parent.named_children()):
            if isinstance(child,conv_type):
                child.update_params()
                if child.has_relu:
                    raise ValueError('Export needs explicit activation handling for this Conv3XC')
                setattr(parent,name,copy.deepcopy(child.eval_conv))
            else: visit(child)
    visit(result)
    return result
