"""Abort-only diagnostics; never rescale loss, gradients or optimizer updates."""
import math
import statistics

def threshold(history):
    return max(1.0,20.0*statistics.median(history)) if len(history)>=1000 else 100.0

def check(value,history):
    limit=threshold(history)
    if not math.isfinite(value) or value>limit:
        raise FloatingPointError('Nonfinite or severe loss anomaly; no automatic rescue')
    return limit

def test():
    assert check(5.6,[])==100
    assert check(.03,[.03]*1000)==1
    for value,history in ((101,[]),(2,[.03]*1000),(float('inf'),[]),(float('nan'),[])):
        try:check(value,history)
        except FloatingPointError:pass
        else:raise AssertionError('Expected anomaly stop')
    print('PASS: finite/extreme/relative loss guard; no optimization modification')

if __name__=='__main__':test()
