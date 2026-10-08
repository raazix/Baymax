"""Resumable official PyTorch downloads for Windows/Python 3.12.

Uses bounded range requests when the large single-file download stalls.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'data' / 'wheels'
FILES = [('torch-2.6.0+cu126-cp312-cp312-win_amd64.whl', 2496082434),
         ('torchvision-0.21.0+cu126-cp312-cp312-win_amd64.whl', 6148507)]

def download(name, size):
    url = 'https://download.pytorch.org/whl/cu126/' + name.replace('+', '%2B')
    destination = DESTINATION / name
    if destination.is_file() and destination.stat().st_size == size:
        print('Already downloaded:', name, flush=True); return
    parts = DESTINATION / (name + '.parts'); parts.mkdir(parents=True, exist_ok=True)
    block_size = 8 * 1024 * 1024
    count = (size + block_size - 1) // block_size
    def fetch(index):
        start = index * block_size; end = min(size - 1, start + block_size - 1)
        expected = end - start + 1
        target = parts / str(index)
        if target.exists() and target.stat().st_size == expected: return expected
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Unexpected server range response')
                    payload = response.read(expected + 1)
                if len(payload) != expected: raise RuntimeError('Incomplete chunk')
                target.write_bytes(payload)
                return expected
            except Exception:
                if attempt == 3: raise
                time.sleep(attempt + 1)
    total = 0; reported = -1
    with ThreadPoolExecutor(max_workers=8) as executor:
        for future in as_completed([executor.submit(fetch, i) for i in range(count)]):
            total += future.result(); percentage = int(total / size * 100)
            if percentage // 10 > reported:
                reported = percentage // 10
                print(f'{name}: {percentage}% ({total // 1024**2} MiB)', flush=True)
    temporary = destination.with_suffix('.assembling')
    digest = hashlib.sha256()
    with temporary.open('wb') as output:
        for i in range(count):
            chunk = (parts / str(i)).read_bytes(); output.write(chunk); digest.update(chunk)
    # Compare against the SHA-256 published in the official package index.
    package = name.split('-')[0]
    with urllib.request.urlopen(f'https://download.pytorch.org/whl/cu126/{package}/', timeout=30) as response:
        index = response.read().decode()
    escaped_name = re.escape(name.replace('+', '%2B'))
    match = re.search(escaped_name + r'#sha256=([a-f0-9]{64})', index, re.IGNORECASE)
    if match is None: raise RuntimeError('Official checksum not found; refusing to install unverified wheel')
    if digest.hexdigest() != match[1].lower(): raise RuntimeError('Wheel checksum mismatch')
    temporary.replace(destination)
    print('Verified official SHA-256:', digest.hexdigest(), flush=True)

if __name__ == '__main__':
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, size in FILES: download(name, size)
