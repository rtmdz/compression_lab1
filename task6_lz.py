import os
import time
import struct
from collections import defaultdict

# ── Part 1: Suffix Array to BWT last column ───────────────────────────────────
def sa_to_bwt(data, sa):
    # for each suffix position, take the char just before it (wrap around)
    return bytes(data[i - 1] if i > 0 else data[-1] for i in sa)

# ── LZ77 ──────────────────────────────────────────────────────────────────────
# tokens: (offset=2B, length=1B, next_char=1B) — 4 bytes per token
# metadata needed: buffer_size (stored in file header)
def lz77_encode(data, buffer_size=255):
    result = bytearray()
    i      = 0
    n      = len(data)
    while i < n:
        win_start = max(0, i - buffer_size)
        window    = data[win_start:i]
        best_off  = 0
        best_len  = 0
        for j in range(len(window)):
            length = 0
            avail  = len(window) - j
            while length < 254 and i + length < n and window[j + length % avail] == data[i + length]:
                length += 1
            if length > best_len:
                best_len = length
                best_off = len(window) - j
        if i + best_len < n:
            next_char = data[i + best_len]
            result   += struct.pack('>HBB', best_off, best_len, next_char)
            i        += best_len + 1
        else:
            # end of data — emit a token with no next_char
            result += struct.pack('>HBB', best_off, best_len, 0)
            i      += best_len + 1
    return bytes(result)

def lz77_decode(data, buffer_size=255):
    result = bytearray()
    i      = 0
    while i + 3 < len(data):
        off, length, next_char = struct.unpack('>HBB', data[i:i+4])
        i += 4
        if off > 0 and length > 0:
            start = len(result) - off
            for k in range(length):
                result.append(result[start + k % off])
        result.append(next_char)
    # remove trailing zero added at end-of-data if present
    return bytes(result)

# ── LZSS ──────────────────────────────────────────────────────────────────────
# improvement over LZ77: use a flag bit per item
# flag=0 → 1 literal byte,  flag=1 → (offset 2B, length 1B) reference
# groups of 8 items packed with a 1-byte flags control header
LZSS_MIN = 3  # only emit reference if match >= 3 bytes

def lzss_encode(data, buffer_size=4096):
    result = bytearray()
    i      = 0
    n      = len(data)
    while i < n:
        flags = 0
        block = bytearray()
        for bit in range(8):
            if i >= n:
                break
            win_start = max(0, i - buffer_size)
            window    = data[win_start:i]
            best_off  = 0
            best_len  = 0
            for j in range(len(window)):
                length = 0
                avail  = len(window) - j
                while length < 255 and i + length < n and window[j + length % avail] == data[i + length]:
                    length += 1
                if length > best_len:
                    best_len = length
                    best_off = len(window) - j
            if best_len >= LZSS_MIN:
                flags |= (1 << bit)
                block += struct.pack('>HB', best_off, best_len)
                i     += best_len
            else:
                block.append(data[i])
                i += 1
        result.append(flags)
        result += block
    return bytes(result)

def lzss_decode(data, buffer_size=4096):
    result = bytearray()
    i      = 0
    while i < len(data):
        flags = data[i]; i += 1
        for bit in range(8):
            if i >= len(data):
                break
            if flags & (1 << bit):
                off, length = struct.unpack('>HB', data[i:i+3]); i += 3
                start = len(result) - off
                for k in range(length):
                    result.append(result[start + k % off])
            else:
                result.append(data[i]); i += 1
    return bytes(result)

# ── LZ78 ──────────────────────────────────────────────────────────────────────
# builds dictionary of seen phrases on the fly
# tokens: (dict_index 2B, next_char 1B) = 3 bytes per token
# metadata needed: max_dict_size
# initial dict NOT stored — both sides start with empty dict
def lz78_encode(data, max_dict=65535):
    dictionary = {b'': 0}
    result     = bytearray()
    phrase     = b''
    for byte in data:
        extended = phrase + bytes([byte])
        if extended in dictionary:
            phrase = extended
        else:
            result += struct.pack('>HB', dictionary[phrase], byte)
            if len(dictionary) < max_dict:
                dictionary[extended] = len(dictionary)
            phrase = b''
    if phrase:
        result += struct.pack('>HB', dictionary[phrase], 0)
    # prepend original length so decoder knows where to stop
    return len(data).to_bytes(4, 'big') + bytes(result)

def lz78_decode(data, max_dict=65535, orig_len=None):
    orig_len   = int.from_bytes(data[:4], 'big')  # read stored length
    data       = data[4:]
    dictionary = {0: b''}
    result     = bytearray()
    i          = 0
    while i + 2 < len(data):
        idx, byte = struct.unpack('>HB', data[i:i+3]); i += 3
        phrase    = dictionary.get(idx, b'') + bytes([byte])
        result   += phrase
        if len(dictionary) < max_dict:
            dictionary[len(dictionary)] = phrase
    return bytes(result[:orig_len])  # trim to original length

# ── LZW ───────────────────────────────────────────────────────────────────────
# improvement over LZ78: initial dict pre-filled with all 256 bytes
# tokens: just a dict index (2B) — no next_char needed
# decoder does NOT need initial dict saved — it's always the same 256 bytes
def lzw_encode(data, max_dict=65535):
    dictionary = {bytes([i]): i for i in range(256)}
    result     = bytearray()
    phrase     = b''
    for byte in data:
        extended = phrase + bytes([byte])
        if extended in dictionary:
            phrase = extended
        else:
            result += struct.pack('>H', dictionary[phrase])
            if len(dictionary) < max_dict:
                dictionary[extended] = len(dictionary)
            phrase = bytes([byte])
    if phrase:
        result += struct.pack('>H', dictionary[phrase])
    return bytes(result)

def lzw_decode(data, max_dict=65535):
    dictionary = {i: bytes([i]) for i in range(256)}
    result     = bytearray()
    i          = 0
    prev       = None
    while i + 1 < len(data):
        code  = struct.unpack('>H', data[i:i+2])[0]; i += 2
        if code in dictionary:
            entry = dictionary[code]
        else:
            entry = prev + prev[:1]  # special case
        result += entry
        if prev is not None and len(dictionary) < max_dict:
            dictionary[len(dictionary)] = prev + entry[:1]
        prev = entry
    return bytes(result)

# ── File encode/decode with header ───────────────────────────────────────────
# header: algo(4B) + orig_size(8B) + buffer_size(4B) + max_dict(4B) = 20 bytes
def encode_file(src, dst, algorithm='lzss', buffer_size=4096, max_dict=65535):
    data = open(src, 'rb').read()
    t0   = time.time()
    if   algorithm == 'lz77': enc = lz77_encode(data, buffer_size)
    elif algorithm == 'lzss': enc = lzss_encode(data, buffer_size)
    elif algorithm == 'lz78': enc = lz78_encode(data, max_dict)
    elif algorithm == 'lzw':  enc = lzw_encode(data,  max_dict)
    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(algorithm.encode().ljust(4)[:4])
        f.write(len(data).to_bytes(8, 'big'))
        f.write(buffer_size.to_bytes(4, 'big'))
        f.write(max_dict.to_bytes(4, 'big'))
        f.write(enc)
    return len(data), os.path.getsize(dst), elapsed

# ── Unit tests ────────────────────────────────────────────────────────────────
def run_unit_tests():
    print("\n=== Task 6: LZ Algorithms ===")

    # sa_to_bwt
    bwt = sa_to_bwt(b'banana', [5, 3, 1, 0, 4, 2])
    print(f"  sa_to_bwt('banana') = {bwt}")
    assert len(bwt) == 6
    print("  ok - sa_to_bwt")

    # LZ77
    for test in [b'aabcaabcabc', b'hello world', b'aaaaaaaaaa', b'abcd']:
        enc = lz77_encode(test)
        dec = lz77_decode(enc)
        assert dec[:len(test)] == test, f"LZ77 fail: got {dec} expected {test}"
    print("  ok - LZ77 roundtrip")

    # LZSS
    for test in [b'aabcaabcabc', b'hello world', b'aaaaaaaaaa', bytes(range(64))]:
        enc = lzss_encode(test)
        dec = lzss_decode(enc)
        assert dec == test, f"LZSS fail: {test[:20]}"
    print("  ok - LZSS roundtrip")

    # LZ78
    for test in [b'aabcaabcabc', b'hello world', b'aaaaaaaaaa']:
        enc = lz78_encode(test)
        dec = lz78_decode(enc)
        assert dec == test, f"LZ78 fail: {test}"
    print("  ok - LZ78 roundtrip")

    # LZW
    for test in [b'aabcaabcabc', b'hello world', b'aaaaaaaaaa', bytes(range(128))]:
        enc = lzw_encode(test)
        dec = lzw_decode(enc)
        assert dec == test, f"LZW fail: {test}"
    print("  ok - LZW roundtrip")

    print("  all tests passed")
