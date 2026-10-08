"""Download official pretrained ResNet18, verifying filename SHA256 prefix."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
DESTINATION = ROOT / 'models/patchcore/backbone/resnet18-f37072fd.pth'


def download():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30) as response:
        size = int(response.headers['Content-Length'])
    print(f'Official backbone size: {size} bytes', flush=True)
    if not DESTINATION.is_file() or DESTINATION.stat().st_size != size:
        parts = DESTINATION.parent / 'download-parts'
        parts.mkdir(exist_ok=True)
        block = 4*1024*1024
        def fetch(index):
            start, end = index*block, min(size-1, (index+1)*block-1)
            target = parts / str(index)
            expected = end-start+1
            if target.is_file() and target.stat().st_size == expected:
                return
            for attempt in range(4):
                try:
                    request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise RuntimeError('Unexpected range response')
                        content = response.read(expected+1)
                    if len(content) != expected:
                        raise RuntimeError('Incomplete range')
                    target.write_bytes(content)
                    print(f'Backbone downloaded block {index}', flush=True)
                    return
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(attempt+1)
        count = (size+block-1)//block
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(fetch, range(count)))
        temporary = DESTINATION.with_suffix('.assembling')
        with temporary.open('wb') as output:
            for index in range(count):
                output.write((parts / str(index)).read_bytes())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if not digest.startswith('f37072fd'):
            raise RuntimeError('Official backbone checksum prefix mismatch')
        temporary.replace(DESTINATION)
    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    if not digest.startswith('f37072fd'):
        raise RuntimeError('Existing backbone checksum prefix mismatch')
    record = {'url': URL, 'size_bytes': size, 'sha256': digest, 'official_hash_prefix': 'f37072fd',
              'verification': 'Official filename SHA256 prefix verified; full digest recorded locally'}
    (DESTINATION.parent / 'provenance.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    return DESTINATION


if __name__ == '__main__':
    download()
