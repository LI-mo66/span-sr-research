"""Pure CPU fixtures for continuation-manager stopping conditions."""
import copy
from continue_to300 import quality_stop

reference={n+suffix:dict(psnr=30.,ssim=.9) for n in
           ('DIV2K_valid','Set5','Set14','B100','Urban100','Manga109') for suffix in ('','_x8')}


def row(changes):
    value=copy.deepcopy(reference)
    for name,delta in changes.items():value[name]['psnr']+=delta
    return dict(metrics=value)


assert quality_stop([row({})],reference) is None
# The supplied296K Manga drop warrants review but does not pass the predeclared stop line.
assert quality_stop([row({'Manga109':-.12266085,'Manga109_x8':-.0874744})],reference) is None
assert quality_stop([row({'B100_x8':-.21})],reference)['rule'].startswith('single_')
assert quality_stop([row({'DIV2K_valid':-.06}),row({'DIV2K_valid_x8':-.06})],reference)['rule'].startswith('two_consecutive_DIV')
assert quality_stop([row({'Manga109':-.16}),row({'Manga109_x8':-.11})],reference)['rule'].startswith('two_consecutive_Manga')
assert quality_stop([row({'Manga109':-.16}),row({'Manga109_x8':-.05})],reference) is None
print('PASS: normal/current296K, immediate-drop, consecutive validation/Manga and recovery cases')
