"""Safely copy the supplied export to ignored storage, preserving existing files."""
import argparse
import hashlib
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def prepare(archive: Path, target: Path):
    with zipfile.ZipFile(archive) as zipped:
        members, destinations = [], set()
        for info in zipped.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or '..' in path.parts or len(path.parts) != 2
                or path.parts[0] != 'processed_data_export' or '\\' in info.filename
                or stat.S_ISLNK(info.external_attr >> 16) or info.file_size > 300_000_000):
                raise ValueError('Unsafe archive member.')
            if info.is_dir():
                continue
            dest = target/path.name
            if dest in destinations or dest.is_symlink():
                raise ValueError('Duplicate or symlink destination.')
            destinations.add(dest)
            members.append((info, dest))
        if sum(i.file_size for i,d in members) > 600_000_000:
            raise ValueError('Archive exceeds the extraction size limit.')
        # Preflight every member before writing any file.
        for info, dest in members:
            if dest.exists():
                with zipped.open(info) as stream:
                    incoming = hashlib.file_digest(stream, 'sha256').hexdigest()
                with dest.open('rb') as stream:
                    existing = hashlib.file_digest(stream, 'sha256').hexdigest()
                if incoming != existing:
                    raise FileExistsError(f'Existing export differs: {dest}; use a new target.')
        target.mkdir(parents=True, exist_ok=True)
        for info, dest in members:
            if not dest.exists():
                with zipped.open(info) as source, dest.open('xb') as output:
                    shutil.copyfileobj(source, output)
        print(f'Verified {len(members)} members in {target.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('local_inputs/processed_data_export.zip'))
    parser.add_argument('--target', type=Path, default=Path('data/processed/export'))
    args = parser.parse_args()
    prepare(args.archive, args.target)

if __name__ == '__main__':
    main()
