import math
import heapq
import struct
import os
from collections import Counter

# ── Entropy calculation ───────────────────────────────────────────────────────
# entropy = -sum(p * log2(p)) for each symbol
# Ms: symbol size in bytes (1=single byte, 2=pairs, etc.)
def calc_entropy(data, Ms=1):
    if not data or len(data) % Ms != 0:
        return 0.0
    symbols = [bytes(data[i:i+Ms]) for i in range(0, len(data), Ms)]
    total   = len(symbols)
    counts  = Counter(symbols)
    entropy = 0.0
    for count in counts.values():
        p        = count / total
        entropy -= p * math.log2(p)
    return entropy

# filter out non-ascii chars (unicode > 127)
def filter_ascii(data):
    return bytes(b for b in data if b <= 127)

# ── MTF (Move-To-Front) ───────────────────────────────────────────────────────
# keeps a list of symbols, when we see a symbol we output its index then move it to front
def mtf_encode(data):
    alphabet = list(range(256))  # [0, 1, 2, ..., 255]
    result   = bytearray()
    for byte in data:
        idx = alphabet.index(byte)   # find position
        result.append(idx)           # output the index
        alphabet.pop(idx)            # move to front
        alphabet.insert(0, byte)
    return bytes(result)

def mtf_decode(data):
    alphabet = list(range(256))
    result   = bytearray()
    for idx in data:
        byte = alphabet[idx]         # get symbol at this index
        result.append(byte)
        alphabet.pop(idx)            # move to front
        alphabet.insert(0, byte)
    return bytes(result)

# ── Huffman coding ────────────────────────────────────────────────────────────
# build frequency table then tree then codes

class HuffNode:
    def __init__(self, symbol, freq, left=None, right=None):
        self.symbol = symbol
        self.freq   = freq
        self.left   = left
        self.right  = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq  = Counter(data)
    heap  = [HuffNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        heapq.heappush(heap, HuffNode(None, l.freq + r.freq, l, r))
    return heap[0] if heap else None

def build_codes(node, prefix='', codes={}):
    if node is None:
        return {}
    if node.symbol is not None:
        codes[node.symbol] = prefix or '0'  # single symbol edge case
    else:
        build_codes(node.left,  prefix + '0', codes)
        build_codes(node.right, prefix + '1', codes)
    return codes

def huffman_encode(data):
    if not data:
        return b'', {}
    codes    = build_codes(build_huffman_tree(data), codes={})
    bitstr   = ''.join(codes[b] for b in data)
    # pad to multiple of 8
    padding  = (8 - len(bitstr) % 8) % 8
    bitstr  += '0' * padding
    encoded  = bytes(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return encoded, codes, padding

def huffman_decode(encoded, codes, padding, original_len):
    # reverse the codes table
    rev   = {v: k for k, v in codes.items()}
    bits  = ''.join(f'{b:08b}' for b in encoded)
    bits  = bits[:len(bits) - padding]  # remove padding
    result = bytearray()
    cur    = ''
    for bit in bits:
        cur += bit
        if cur in rev:
            result.append(rev[cur])
            cur = ''
            if len(result) == original_len:
                break
    return bytes(result)

# file I/O for huffman
# header: MAGIC(4) + original_len(8) + padding(1) + num_codes(2) + codes + data
HUFF_MAGIC = b'HUF1'

def huffman_encode_file(src, dst):
    data              = open(src, 'rb').read()
    encoded, codes, padding = huffman_encode(data)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(HUFF_MAGIC)
        f.write(len(data).to_bytes(8, 'big'))   # original size
        f.write(bytes([padding]))                # padding bits
        # write codes table: [symbol(1)][code_len(1)][code bits packed]
        f.write(len(codes).to_bytes(2, 'big'))
        for sym, code in codes.items():
            f.write(bytes([sym]))
            f.write(bytes([len(code)]))
            # pack code bits into bytes
            padded = code + '0' * ((8 - len(code) % 8) % 8)
            for i in range(0, len(padded), 8):
                f.write(bytes([int(padded[i:i+8], 2)]))
        f.write(encoded)
    return len(data), os.path.getsize(dst)

def huffman_decode_file(src, dst):
    with open(src, 'rb') as f:
        assert f.read(4) == HUFF_MAGIC
        orig_len  = int.from_bytes(f.read(8), 'big')
        padding   = f.read(1)[0]
        num_codes = int.from_bytes(f.read(2), 'big')
        codes     = {}
        for _ in range(num_codes):
            sym      = f.read(1)[0]
            code_len = f.read(1)[0]
            nb       = math.ceil(code_len / 8)
            raw      = f.read(nb)
            bits     = ''.join(f'{b:08b}' for b in raw)
            codes[sym] = bits[:code_len]
        encoded = f.read()
    decoded = huffman_decode(encoded, codes, padding, orig_len)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    open(dst, 'wb').write(decoded)
    return len(encoded), len(decoded)

# ── Arithmetic coding ─────────────────────────────────────────────────────────
# encode a byte string into a float in [0, 1)
# returns the encoded float and the probability model used
def arithmetic_encode(data):
    if not data:
        return 0.0, {}
    freq    = Counter(data)
    total   = len(data)
    symbols = sorted(freq.keys())
    # build cumulative probability table
    probs   = {}
    cum     = 0.0
    for sym in symbols:
        p          = freq[sym] / total
        probs[sym] = (cum, cum + p)
        cum       += p
    low, high = 0.0, 1.0
    for byte in data:
        lo, hi = probs[byte]
        rng    = high - low
        high   = low + rng * hi
        low    = low + rng * lo
    return (low + high) / 2, probs

def arithmetic_decode(value, probs, length):
    # reverse lookup: find symbol whose range contains value
    result = bytearray()
    for _ in range(length):
        for sym, (lo, hi) in probs.items():
            if lo <= value < hi:
                result.append(sym)
                value = (value - lo) / (hi - lo)
                break
    return bytes(result)

# find when boundaries collapse (precision loss in double)
def find_precision_limit():
    import random
    rng  = random.Random(42)
    data = bytes([rng.randint(0, 3) for _ in range(1000)])  # small alphabet
    freq = Counter(data)
    tot  = len(data)
    syms = sorted(freq.keys())
    probs = {}
    cum   = 0.0
    for s in syms:
        p = freq[s] / tot
        probs[s] = (cum, cum + p)
        cum      += p
    low, high = 0.0, 1.0
    for i, byte in enumerate(data):
        lo, hi = probs[byte]
        rng_   = high - low
        high   = low + rng_ * hi
        low    = low + rng_ * lo
        if low >= high:
            return i  # boundaries collapsed at this index
    return len(data)  # never collapsed

def run_unit_tests():
    print("\n=== Task 3: Entropy + MTF + Huffman + Arithmetic ===")

    # entropy
    data = b'aaabbc'
    e    = calc_entropy(data, Ms=1)
    assert 0 < e < 2, f"entropy out of range: {e}"
    print(f"  ok - entropy of 'aaabbc' = {e:.4f} bits/symbol")

    # MTF roundtrip
    for test in [b'hello world', b'\x00' * 50, bytes(range(256))]:
        assert mtf_decode(mtf_encode(test)) == test
    print("  ok - MTF encode/decode roundtrip")

    # Huffman roundtrip
    for test in [b'hello world', b'aaabbbcccc', bytes(range(256))]:
        enc, codes, pad = huffman_encode(test)
        dec = huffman_decode(enc, codes, pad, len(test))
        assert dec == test, f"huffman fail: {test}"
    print("  ok - Huffman encode/decode roundtrip")

    # Arithmetic
    test    = b'abcd'
    val, pr = arithmetic_encode(test)
    dec     = arithmetic_decode(val, pr, len(test))
    assert dec == test
    print("  ok - Arithmetic encode/decode roundtrip")

    limit = find_precision_limit()
    print(f"  ok - Arithmetic precision limit at string length ~{limit}")

    print("  all tests passed")
