import os
from collections import Counter
from task2_rle import rle_encode_general, rle_decode_general

# ── BWT forward (direct matrix) ───────────────────────────────────────────────
# builds all rotations, sorts them, returns last column + original row index
def bwt_encode(data):
    n         = len(data)
    rotations = sorted(data[i:] + data[:i] for i in range(n))
    last_col  = bytes(r[-1] for r in rotations)
    idx       = rotations.index(data)  # index of original string in sorted matrix
    return last_col, idx

# ── BWT inverse (direct matrix rebuild) ───────────────────────────────────────
def bwt_decode_direct(last_col, idx):
    n    = len(last_col)
    mat  = [b'' for _ in range(n)]
    for _ in range(n):
        mat = sorted(bytes([last_col[i]]) + mat[i] for i in range(n))
    return mat[idx]

# ── BWT inverse (fast: using shift+sort permutation + counting sort) ──────────
# much faster than rebuilding full matrix
def counting_sort(data):
    # stable counting sort on bytes — O(n)
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    result = bytearray()
    for val in range(256):
        result.extend(bytes([val]) * counts[val])
    return bytes(result)

def bwt_decode_fast(last_col, idx):
    n         = len(last_col)
    first_col = counting_sort(last_col)   # sort last col to get first col
    # build the permutation T such that T[i] = position in first_col
    # that corresponds to last_col[i]
    counts = [0] * 256
    T      = [0] * n
    # count occurrences in first column up to each position
    first_positions = {}
    pos = 0
    for val in range(256):
        first_positions[val] = pos
        pos += sum(1 for b in last_col if b == val)
    cur_count = [0] * 256
    for i in range(n):
        b    = last_col[i]
        T[i] = first_positions[b] + cur_count[b]
        cur_count[b] += 1
    # follow the permutation to reconstruct
    result = bytearray(n)
    i      = idx
    for j in range(n - 1, -1, -1):
        result[j] = last_col[i]
        i         = T[i]
    return bytes(result)

# ── Block BWT (for large files) ───────────────────────────────────────────────
# direct BWT is O(n^2) space — too much for large files, so we split into blocks
BLOCK_SIZE = 10_000  # 100 KB blocks

BWT_MAGIC = b'BWT1'
# file format: MAGIC(4) + num_blocks(4) + [block_size(4) + idx(4) + last_col_data] * n

def bwt_encode_file(src, dst):
    data = open(src, 'rb').read()
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    blocks = [data[i:i+BLOCK_SIZE] for i in range(0, len(data), BLOCK_SIZE)]
    with open(dst, 'wb') as f:
        f.write(BWT_MAGIC)
        f.write(len(blocks).to_bytes(4, 'big'))
        for block in blocks:
            last_col, idx = bwt_encode(block)
            # apply RLE on the BWT output (BWT groups similar bytes together)
            rle_data = rle_encode_general(last_col, Ms=1, Mc=1)
            f.write(len(rle_data).to_bytes(4, 'big'))
            f.write(len(block).to_bytes(4, 'big'))   # original block size
            f.write(idx.to_bytes(4, 'big'))
            f.write(rle_data)
    return len(data), os.path.getsize(dst)

def bwt_decode_file(src, dst):
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
            block      = bwt_decode_fast(last_col, idx)
            result.extend(block)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    open(dst, 'wb').write(result)
    return os.path.getsize(src), len(result)

def run_unit_tests():
    print("\n=== Task 4: BWT ===")

    # test on "banana" = 0x62 0x61 0x6e 0x61 0x6e 0x61
    banana   = bytes([0x62, 0x61, 0x6e, 0x61, 0x6e, 0x61])
    last, idx = bwt_encode(banana)
    print(f"  BWT of 'banana': last_col={last.hex()}, idx={idx}")

    # verify inverse (direct)
    restored = bwt_decode_direct(last, idx)
    assert restored == banana, f"direct decode fail: {restored}"
    print("  ok - BWT direct decode")

    # verify inverse (fast)
    restored2 = bwt_decode_fast(last, idx)
    assert restored2 == banana, f"fast decode fail: {restored2}"
    print("  ok - BWT fast decode (counting sort)")

    # roundtrip on various inputs
    for test in [b'hello world', b'aaabbbccc', bytes(range(128))]:
        last, idx = bwt_encode(test)
        assert bwt_decode_fast(last, idx) == test
    print("  ok - BWT roundtrip on various inputs")

    # file roundtrip
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data = b'hello world ' * 200 + b'banana ' * 100
        open(f'{tmp}/orig.bin', 'wb').write(data)
        bwt_encode_file(f'{tmp}/orig.bin', f'{tmp}/enc.bwt')
        bwt_decode_file(f'{tmp}/enc.bwt',  f'{tmp}/dec.bin')
        assert open(f'{tmp}/dec.bin', 'rb').read() == data
    print("  ok - BWT file encode/decode roundtrip with blocks")

    print("  all tests passed")
