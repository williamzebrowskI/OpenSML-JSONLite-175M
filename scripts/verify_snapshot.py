#!/usr/bin/env python3
"""Verify every published snapshot file against its SHA-256 manifest."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'provenance/SNAPSHOT_MANIFEST.json').read_text())
    for name,record in manifest['files'].items():
        path=ROOT/name
        if path.is_symlink() or not path.is_file():raise ValueError('Missing/nonregular file: '+name)
        raw=path.read_bytes()
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('Snapshot integrity mismatch: '+name)
    print('Verified',len(manifest['files']),'snapshot files; model weights are not in this Git snapshot.')
if __name__=='__main__':main()
