"""Decode bounded JPEG/PNG uploads before running any inference."""
import struct
import cv2
import numpy as np

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000

def dimensions(content: bytes):
    # Read dimensions before decoding to prevent huge decompressed allocations.
    if content.startswith(b'\x89PNG\r\n\x1a\n') and len(content) >= 24:
        width, height = struct.unpack('>II', content[16:24]); media = 'image/png'
    elif content.startswith(b'\xff\xd8'):
        position = 2; width = height = 0; media = 'image/jpeg'
        while position < len(content):
            if content[position] != 255: raise ValueError('Malformed JPEG marker')
            while position < len(content) and content[position] == 255: position += 1
            if position >= len(content): break
            marker = content[position]; position += 1
            if marker in [0xD8, 0xD9, 0x01] or 0xD0 <= marker <= 0xD7: continue
            if position + 2 > len(content): break
            length = int.from_bytes(content[position:position+2], 'big')
            if length < 2 or position + length > len(content): raise ValueError('Malformed JPEG segment')
            if marker in [0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF]:
                if length < 8: raise ValueError('Malformed JPEG dimensions')
                height, width = struct.unpack('>HH', content[position+3:position+7]); break
            position += length
        if not width or not height: raise ValueError('JPEG has no valid dimensions')
    else: raise ValueError('Only JPEG and PNG images are supported')
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise OverflowError('Image exceeds valid dimensions or 16 megapixels')
    return width, height, media

def decode(content: bytes):
    if len(content) > MAX_BYTES: raise OverflowError('Frame exceeds 8 MiB limit')
    width, height, media = dimensions(content)
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if frame is None: raise ValueError('Image could not be decoded')
    if frame.shape[:2] != (height, width): raise ValueError('Decoded dimensions differ from image header')
    return frame, media
