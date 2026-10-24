"""Download and verify the versioned research archive using the standard library."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('research_download'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    manifest=json.loads((Path(__file__).parent/'config/revision_release_manifest.json').read_text(encoding='utf-8'))
    archive=manifest['archive']
    args.output.mkdir(parents=True,exist_ok=True)
    base=manifest['repository']+'/releases/download/'+archive.get('release_tag','v'+manifest['version'])+'/'
    def fetch(part):
        dest=args.output/part['name']
        if dest.exists() and dest.stat().st_size==part['bytes'] and digest(dest)==part['sha256']:
            return dest
        temporary=dest.with_suffix(dest.suffix+'.partial')
        with urlopen(base+part['name'],timeout=120) as response,temporary.open('wb') as out:
            shutil.copyfileobj(response,out,1024*1024)
        if temporary.stat().st_size!=part['bytes'] or digest(temporary)!=part['sha256']:
            raise ValueError('Download checksum mismatch: '+part['name'])
        temporary.replace(dest)
        print('Verified',dest.name,flush=True)
        return dest
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        paths=list(pool.map(fetch,archive['parts']))
    target=args.output/archive['name']
    temporary=target.with_suffix('.zip.partial')
    with temporary.open('wb') as out:
        for path in paths:
            with path.open('rb') as source:
                shutil.copyfileobj(source,out,1024*1024)
    if temporary.stat().st_size!=archive['bytes'] or digest(temporary)!=archive['sha256']:
        raise ValueError('Reconstructed archive checksum mismatch')
    temporary.replace(target)
    print('Verified research archive:',target.resolve())
    print('Extract to a short local path, then follow docs/REVISION_REPRODUCTION.md.')

if __name__=='__main__':
    main()
