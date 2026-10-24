"""Read-only final gate before an explicitly authorized instance release."""
import json
from pathlib import Path
import sys
from backup import sha


def successful(status):
    return status.get('success') is True and all(status.get(k)==0 for k in
        ('training_exit_code','comparison_exit_code','plotting_exit_code','backup_exit_code','exit_code'))


def validate(path):
    status=json.loads(path.read_text())
    if not successful(status):raise ValueError('Incomplete or failed run; instance retained')
    destination=Path(status['backup']).resolve()
    if not destination.is_relative_to(Path('/home/featurize/work')):
        raise ValueError('Verified backup must be persistent')
    if not (destination/'BACKUP_COMPLETE').is_file():raise ValueError('Backup marker missing')
    info=json.loads((destination/'backup-manifest.json').read_text())
    if info['training_exit_code']!=0 or info['comparison_exit_code']!=0:
        raise ValueError('Backup records a failed run')
    archived=json.loads((destination/'FINAL_STATUS.json').read_text())
    if not successful(archived) or archived['id']!=status['id']:
        raise ValueError('Final archived status mismatch')
    if not info['files']:raise ValueError('Empty backup')
    for name,expected in info['files'].items():
        file=(destination/name).resolve()
        if not file.is_relative_to(destination) or sha(file)!=expected:
            raise ValueError('Backup hash mismatch: '+name)
    return destination


if __name__=='__main__':
    print(validate(Path(sys.argv[1])),flush=True)
