"""All shutdown/permission/GPU calls are mocks. Does not power off any machine."""
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
from bootstrap import ROOT
import power

def main():
    with tempfile.TemporaryDirectory(prefix='strip-power-',dir=ROOT/'tmp') as tmp:
        group=Path(tmp);executable=group/'shutdown';executable.write_text('MOCK ONLY')
        with patch.object(power.os,'geteuid',return_value=0,create=True),patch.object(power.os,'access',return_value=True),patch.object(power.subprocess,'run') as run:
            command=power.plan(str(executable));assert command==[str(executable),'-h','now'];run.assert_not_called()
        with patch.object(power.os,'geteuid',return_value=1000,create=True),patch.object(power.os,'access',return_value=True),patch.object(power.shutil,'which',return_value='/usr/bin/sudo'),patch.object(power.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','permission denied')) as run:
            try:power.plan(str(executable))
            except RuntimeError:pass
            else:raise AssertionError('Denied sudo permission must block before queue')
            assert run.call_args.args[0]==['/usr/bin/sudo','-n','-l',str(executable),'-h','now']
        backup=group/'backup';backup.mkdir();(backup/'BACKUP_COMPLETE').touch();(backup/'backup-manifest.json').write_text('{}')
        returned=group/'return.zip';returned.write_bytes(b'ALREADY VERIFIED BY COLLECT IN REAL QUEUE')
        status=dict(K15=dict(status='COMPLETE',backup=dict(status='HASH_VERIFIED',path=str(backup))),K31=dict(status='HOLD_COST'))
        with patch.object(power.subprocess,'check_output',return_value=''),patch.object(power.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')) as run,patch.object(power.os,'sync',create=True):
            power.finish(group,status,command,returned);run.assert_called_once_with(command,capture_output=True,text=True)
            assert json.loads((group/'shutdown-status.json').read_text())['status']=='COMMAND_ACCEPTED'
        for occupied,broken in [('123, 100 MiB',False),('',True)]:
            if broken:(backup/'BACKUP_COMPLETE').unlink()
            with patch.object(power.subprocess,'check_output',return_value=occupied),patch.object(power.subprocess,'run') as run:
                try:power.finish(group,status,command,returned)
                except RuntimeError:pass
                else:raise AssertionError('Busy GPU or broken backup must block shutdown')
                run.assert_not_called()
        (backup/'BACKUP_COMPLETE').touch()
        status['K15']['status']='FAILED'
        with patch.object(power.subprocess,'check_output',return_value=''),patch.object(power.subprocess,'run',return_value=subprocess.CompletedProcess([],7,'','mock shutdown failed')),patch.object(power.os,'sync',create=True):
            try:power.finish(group,status,command,returned)
            except RuntimeError:pass
            else:raise AssertionError('Shutdown failure must be visible')
            assert json.loads((group/'shutdown-status.json').read_text())['status']=='FAILED'
        print('POWER MOCK PASS: root plan does not execute, denied sudo blocks, verified backup precedes shutdown, busy GPU/incomplete backup blocks, command failure recorded')

if __name__=='__main__':main()
