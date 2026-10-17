"""Download only the pinned official release, atomically and with SHA256."""
import argparse
from pathlib import Path
import urllib.request
from config import WEIGHT_SHA256
from teacher import WEIGHT_NAME, WEIGHT_URL, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / WEIGHT_NAME
    if target.exists():
        if sha256(target) != WEIGHT_SHA256:
            raise ValueError(f'Existing weight differs; preserved: {target}')
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(dir=args.directory, suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            urllib.request.urlretrieve(WEIGHT_URL, temporary)
            if sha256(temporary) != WEIGHT_SHA256:
                raise ValueError('Official download hash mismatch')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print(target.resolve())


if __name__ == '__main__':
    main()
