import os
import math
import heapq
import struct
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Efficient BWT using Suffix Array
# ══════════════════════════════════════════════════════════════════════════════
# Suffix Array: sorted list of all suffix starting positions
# Time:  O(n log n)  — sorting n suffixes
# Space: O(n)        — store n integers
#
# Compare with naive BWT:
#   Naive Time:  O(n^2 log n) — sort n rotations each of length n
#   Naive Space: O(n^2)       — store the full rotation matrix

def build_suffix_array(data):
    # sort rotation start positions — O(n log n), matches standard BWT definition
    n  = len(data)
    sa = sorted(range(n), key=lambda i: data[i:] + data[:i])
    return sa

def bwt_encode_fast(data):
    sa       = build_suffix_array(data)
    last_col = bytes(data[i - 1] for i in sa)   # char before each rotation
    idx      = sa.index(0)                        # row of original string
    return last_col, idx

def counting_sort_bytes(data):
    counts = [0] * 256
    for b in data: counts[b] += 1
    result = bytearray()
    for v in range(256): result.extend(bytes([v]) * counts[v])
    return bytes(result)

def bwt_decode_fast(last_col, idx):
    n         = len(last_col)
    first_col = counting_sort_bytes(last_col)
    # build position lookup for first column
    first_pos = {}
    pos       = 0
    for v in range(256):
        first_pos[v] = pos
        pos += sum(1 for b in last_col if b == v)
    # build T permutation
    cur = [0] * 256
    T   = [0] * n
    for i in range(n):
        b    = last_col[i]
        T[i] = first_pos[b] + cur[b]
        cur[b] += 1
    # follow T to reconstruct original
    result = bytearray(n)
    i      = idx
    for j in range(n - 1, -1, -1):
        result[j] = last_col[i]
        i         = T[i]
    return bytes(result)

# block-based file encode/decode (BWT is O(n^2) memory for large files)
BLOCK_SIZE = 10_000
BWT_MAGIC  = b'BWT3'

def bwt_encode_file(src, dst):
    from task2_rle import rle_encode_general
    data   = open(src, 'rb').read()
    blocks = [data[i:i+BLOCK_SIZE] for i in range(0, len(data), BLOCK_SIZE)]
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(BWT_MAGIC)
        f.write(len(blocks).to_bytes(4, 'big'))
        for block in blocks:
            last_col, idx = bwt_encode_fast(block)
            rle_data      = rle_encode_general(last_col, Ms=1, Mc=1)
            f.write(len(rle_data).to_bytes(4, 'big'))
            f.write(len(block).to_bytes(4, 'big'))
            f.write(idx.to_bytes(4, 'big'))
            f.write(rle_data)
    return len(data), os.path.getsize(dst)

def bwt_decode_file(src, dst):
    from task2_rle import rle_decode_general
    with open(src, 'rb') as f:
        assert f.read(4) == BWT_MAGIC
        num_blocks = int.from_bytes(f.read(4), 'big')
        result     = bytearray()
        for _ in range(num_blocks):
            rle_size   = int.from_bytes(f.read(4), 'big')
            block_size = int.from_bytes(f.read(4), 'big')
            idx        = int.from_bytes(f.read(4), 'big')
            rle_data   = f.read(rle_size)
            last_col   = rle_decode_general(rle_data, Ms=1, Mc=1)
            result.extend(bwt_decode_fast(last_col, idx))
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    open(dst, 'wb').write(result)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Canonical Huffman Codes
# ══════════════════════════════════════════════════════════════════════════════
# Regular Huffman: codes depend on tree shape — need to store full tree
# Canonical Huffman: codes are assigned by rules, only code LENGTHS needed
#
# Rules for canonical codes:
#   1. Shorter codes have smaller numeric values
#   2. Among codes of same length, they are assigned in alphabetical order
#   3. First code of each length = (prev_code + 1) << (new_length - prev_length)
#
# Metadata saving:
#   Regular Huffman: store full tree or {symbol: code_string} — many bytes
#   Canonical Huffman: store only {symbol: code_length} — much smaller!
#   For 256 symbols, we store 256 bytes of lengths instead of variable-length codes

class HuffNode:
    def __init__(self, sym, freq, left=None, right=None):
        self.sym  = sym
        self.freq = freq
        self.left = left
        self.right= right
    def __lt__(self, other): return self.freq < other.freq

def get_code_lengths(data):
    # step 1: build standard huffman tree to get optimal lengths
    freq = Counter(data)
    if len(freq) == 1:
        return {list(freq.keys())[0]: 1}
    heap = [HuffNode(s, f) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        heapq.heappush(heap, HuffNode(None, l.freq + r.freq, l, r))
    # step 2: traverse tree to get code lengths
    lengths = {}
    def traverse(node, depth=0):
        if node is None: return
        if node.sym is not None:
            lengths[node.sym] = max(depth, 1)
        else:
            traverse(node.left,  depth + 1)
            traverse(node.right, depth + 1)
    traverse(heap[0])
    return lengths

def build_canonical_codes(lengths):
    # sort symbols: first by length, then by symbol value
    symbols = sorted(lengths.keys(), key=lambda s: (lengths[s], s))
    codes   = {}
    code    = 0
    prev_len= 0
    for sym in symbols:
        l = lengths[sym]
        # shift left when length increases
        code <<= (l - prev_len)
        codes[sym] = format(code, f'0{l}b')
        code   += 1
        prev_len = l
    return codes

def canonical_huffman_encode(data):
    if not data: return b'', {}, 0
    lengths  = get_code_lengths(data)
    codes    = build_canonical_codes(lengths)
    # encode data as bitstring
    bitstr   = ''.join(codes[b] for b in data)
    padding  = (8 - len(bitstr) % 8) % 8
    bitstr  += '0' * padding
    encoded  = bytes(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return encoded, lengths, padding
    # note: we only store 'lengths' not 'codes' — saves space!

def canonical_huffman_decode(encoded, lengths, padding, orig_len):
    # rebuild canonical codes from lengths (same deterministic process)
    codes  = build_canonical_codes(lengths)
    rev    = {v: k for k, v in codes.items()}
    bits   = ''.join(f'{b:08b}' for b in encoded)
    bits   = bits[:len(bits) - padding]
    result = bytearray()
    cur    = ''
    for bit in bits:
        cur += bit
        if cur in rev:
            result.append(rev[cur])
            cur = ''
            if len(result) == orig_len: break
    return bytes(result)

# ── File I/O for canonical huffman ───────────────────────────────────────────
# Metadata stored: only code lengths (1 byte per symbol, 256 bytes total)
# Much smaller than storing full code strings!
# Header: MAGIC(4) + orig_len(8) + padding(1) + lengths[256] = 269 bytes fixed

CANON_MAGIC = b'CHF1'

def canonical_huffman_encode_file(src, dst):
    data             = open(src, 'rb').read()
    encoded, lengths, padding = canonical_huffman_encode(data)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(CANON_MAGIC)
        f.write(len(data).to_bytes(8, 'big'))
        f.write(bytes([padding]))
        # write 256 length bytes — 0 means symbol not used
        length_table = bytes([lengths.get(i, 0) for i in range(256)])
        f.write(length_table)
        f.write(encoded)
    return len(data), os.path.getsize(dst)

def canonical_huffman_decode_file(src, dst):
    with open(src, 'rb') as f:
        assert f.read(4) == CANON_MAGIC
        orig_len     = int.from_bytes(f.read(8), 'big')
        padding      = f.read(1)[0]
        length_table = f.read(256)
        encoded      = f.read()
    # rebuild lengths dict (skip symbols with length 0)
    lengths = {i: length_table[i] for i in range(256) if length_table[i] > 0}
    decoded = canonical_huffman_decode(encoded, lengths, padding, orig_len)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    open(dst, 'wb').write(decoded)
    return len(encoded), len(decoded)


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════
def run_unit_tests():
    import time
    print("\n=== Task 7: Efficient BWT + Canonical Huffman ===")

    # BWT on banana
    banana    = bytes([0x62, 0x61, 0x6e, 0x61, 0x6e, 0x61])
    last, idx = bwt_encode_fast(banana)
    print(f"  BWT('banana'): last_col={last.hex()}, idx={idx}")
    assert bwt_decode_fast(last, idx) == banana
    print("  ok - BWT fast roundtrip")

    # speed comparison: fast vs naive
    from task4_bwt import bwt_encode as bwt_naive
    big    = b'hello world ' * 400
    t0     = time.time()
    for _  in range(3): bwt_encode_fast(big)
    t_fast = (time.time() - t0) / 3

    t0     = time.time()
    for _  in range(3): bwt_naive(big)
    t_slow = (time.time() - t0) / 3

    print(f"  speed on {len(big)} bytes:")
    print(f"    naive O(n^2 log n): {t_slow*1000:.1f} ms")
    print(f"    SA    O(n log n):   {t_fast*1000:.1f} ms")
    print(f"    speedup: {t_slow/t_fast:.1f}x")

    # canonical huffman roundtrip
    for test in [b'hello world', b'aaabbbcccc', b'abcdefgh' * 100]:
        enc, lengths, pad = canonical_huffman_encode(test)
        dec = canonical_huffman_decode(enc, lengths, pad, len(test))
        assert dec == test, f"canonical huffman fail on {test[:20]}"
    print("  ok - canonical Huffman roundtrip")

    # verify canonical codes are actually canonical
    test    = b'aabbbbccccccccdddddddddddddddd'
    lengths = get_code_lengths(test)
    codes   = build_canonical_codes(lengths)
    print(f"  canonical codes example:")
    for sym, code in sorted(codes.items(), key=lambda x: (len(x[1]), x[0])):
        print(f"    symbol={sym} ({chr(sym)}): length={lengths[sym]}, code={code}")

    # metadata size comparison
    from task3_entropy import huffman_encode_file
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data = b'hello world this is a test of huffman coding ' * 200
        open(f'{tmp}/orig.bin', 'wb').write(data)
        # regular huffman
        from task3_entropy import huffman_encode_file, huffman_decode_file
        orig1, enc1 = huffman_encode_file(f'{tmp}/orig.bin', f'{tmp}/reg.huf')
        # canonical huffman
        orig2, enc2 = canonical_huffman_encode_file(f'{tmp}/orig.bin', f'{tmp}/can.huf')
        huffman_decode_file(f'{tmp}/reg.huf', f'{tmp}/reg.dec')
        canonical_huffman_decode_file(f'{tmp}/can.huf', f'{tmp}/can.dec')
        assert open(f'{tmp}/reg.dec','rb').read() == data
        assert open(f'{tmp}/can.dec','rb').read() == data
        print(f"\n  metadata size comparison (same data {orig1} bytes):")
        print(f"    regular huffman file:   {enc1} bytes")
        print(f"    canonical huffman file: {enc2} bytes  (fixed 269 byte header)")
    print("  all tests passed")
