"""Isolated output-KD entry; upstream training files are not modified."""
import os
import sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
import trainer
if __name__ == '__main__':
    trainer.main()
