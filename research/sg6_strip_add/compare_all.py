"""Report against fixed SG6 and reviewed K31 separately, one selected candidate weight."""
import argparse
import json
from pathlib import Path
from bootstrap import trainer
from compare import compare

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();audit=json.loads((a.group/'frozen/reuse-audit.json').read_text())
    for name,r in audit['k31_provenance']['files'].items():
        if trainer.digest(Path(audit['k31_reference'])/name)!=r['sha256']:raise ValueError('Protected K31 reference changed')
    sg6=compare(a.group/'frozen',a.run,a.group/'vs-SG6.json')
    alt=dict(audit,reference_run=audit['k31_reference'],
        reference_manifest_sha256=audit['k31_provenance']['files']['manifest.json']['sha256'],
        reference_history_sha256=audit['k31_provenance']['files']['epochs.jsonl']['sha256'])
    folder=a.group/'k31-comparison-input';folder.mkdir(exist_ok=True)
    (folder/'reuse-audit.json').write_text(json.dumps(alt,indent=2))
    k31=compare(folder,a.run,a.group/'vs-K31.json')
    print(json.dumps(dict(status='COMPLETE',selected_update=sg6['candidate_selected_update'],
        vs_SG6=sg6['datasets']['DIV2K_valid'],vs_K31=k31['datasets']['DIV2K_valid']),indent=2))

if __name__=='__main__':main()
