import os
import tempfile

MAGIC = b'RLE1'


# Stage 1 - basic RLE
# format: [count 1 byte][symbol 1 byte]
def rle_encode_basic(data):
    if not data:
        return b''
    result = bytearray()
    i = 0
    while i < len(data):
        sym   = data[i]
        count = 1
        while i + count < len(data) and data[i + count] == sym and count < 255:
            count += 1
        result.append(count)
        result.append(sym)
        i += count
    return bytes(result)


def rle_decode_basic(data):
    result = bytearray()
    i = 0
    while i + 1 < len(data):
        count = data[i]
        sym   = data[i + 1]
        result.extend(bytes([sym]) * count)
        i += 2
    return bytes(result)


# Stage 2 - RLE with MSB flag for literal runs
# bit 7 = 0 -> repeat:  [0b0ccccccc][symbol]
# bit 7 = 1 -> literal: [0b1ccccccc][bytes...]
def rle_encode_msb(data):
    if not data:
        return b''
    result = bytearray()
    i = 0
    n = len(data)
    while i < n:
        sym = data[i]
        run = 1
        while i + run < n and data[i + run] == sym and run < 127:
            run += 1

        if run >= 2:
            result.append(run)
            result.append(sym)
            i += run
        else:
            j = i
            while i < n and (i - j) < 127:
                if i + 1 < n and data[i] == data[i + 1]:
                    break
                i += 1
            lit = i - j
            if lit == 0:
                lit = 1
                i  += 1
            result.append(0x80 | lit)
            result.extend(data[j:j + lit])
    return bytes(result)


def rle_decode_msb(data):
    result = bytearray()
    i = 0
    while i < len(data):
        ctrl = data[i]
        i   += 1
        if ctrl & 0x80:
            count = ctrl & 0x7F
            result.extend(data[i:i + count])
            i += count
        else:
            result.extend(bytes([data[i]]) * ctrl)
            i += 1
    return bytes(result)


# Stage 3 - general RLE with Ms (symbol size) and Mc (control size)
def rle_encode_general(data, Ms=1, Mc=1):
    if len(data) % Ms != 0:
        raise ValueError(f"data length must be divisible by Ms={Ms}")

    msb_mask  = 1 << (Mc * 8 - 1)
    max_count = msb_mask - 1
    symbols   = [bytes(data[i:i + Ms]) for i in range(0, len(data), Ms)]

    result = bytearray()
    i = 0
    while i < len(symbols):
        sym = symbols[i]
        run = 1
        while i + run < len(symbols) and symbols[i + run] == sym and run < max_count:
            run += 1

        if run >= 2:
            result.extend(run.to_bytes(Mc, 'big'))
            result.extend(sym)
            i += run
        else:
            j = i
            while i < len(symbols) and (i - j) < max_count:
                if i + 1 < len(symbols) and symbols[i] == symbols[i + 1]:
                    break
                i += 1
            lit = i - j
            if lit == 0:
                lit = 1
                i  += 1
            result.extend((msb_mask | lit).to_bytes(Mc, 'big'))
            for s in symbols[j:j + lit]:
                result.extend(s)

    return bytes(result)


def rle_decode_general(data, Ms=1, Mc=1):
    msb_mask   = 1 << (Mc * 8 - 1)
    count_mask = msb_mask - 1
    result     = bytearray()
    i          = 0
    while i < len(data):
        ctrl       = int.from_bytes(data[i:i + Mc], 'big')
        i         += Mc
        count      = ctrl & count_mask
        if ctrl & msb_mask:
            result.extend(data[i:i + count * Ms])
            i += count * Ms
        else:
            result.extend(data[i:i + Ms] * count)
            i += Ms
    return bytes(result)


# UTF-8 problem: cyrillic chars are 2 bytes each in UTF-8
# applying Ms=1 splits characters across symbol boundaries
# fix: convert to UTF-32 first so every char is exactly 4 bytes
def utf8_to_utf32(b):
    return b.decode('utf-8').encode('utf-32-le')

def utf32_to_utf8(b):
    return b.decode('utf-32-le').encode('utf-8')


# Stage 4 - file I/O with metadata header
# header: MAGIC(4) + Ms(1) + Mc(1) + original_size(8) = 14 bytes
def rle_encode_file(src, dst, Ms=1, Mc=1):
    data    = open(src, 'rb').read()
    encoded = rle_encode_general(data, Ms, Mc)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(MAGIC)
        f.write(bytes([Ms, Mc]))
        f.write(len(data).to_bytes(8, 'big'))
        f.write(encoded)
    return len(data), len(encoded)


def rle_decode_file(src, dst):
    with open(src, 'rb') as f:
        assert f.read(4) == MAGIC
        Ms        = f.read(1)[0]
        Mc        = f.read(1)[0]
        orig_size = int.from_bytes(f.read(8), 'big')
        encoded   = f.read()
    decoded = rle_decode_general(encoded, Ms, Mc)
    assert len(decoded) == orig_size
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    open(dst, 'wb').write(decoded)
    return len(encoded), len(decoded)


def estimate_rle_ratio(data, Ms=1):
    if not data or len(data) % Ms:
        return 1.0
    symbols = [bytes(data[i:i + Ms]) for i in range(0, len(data), Ms)]
    runs = []
    i = 0
    while i < len(symbols):
        c = 1
        while i + c < len(symbols) and symbols[i + c] == symbols[i]:
            c += 1
        runs.append(c)
        i += c
    est = 0
    j = 0
    while j < len(runs):
        if runs[j] >= 2:
            est += 1 + Ms
            j   += 1
        else:
            lit = 0
            while j < len(runs) and runs[j] == 1 and lit < 127:
                lit += 1
                j   += 1
            est += 1 + lit * Ms
    return len(data) / est if est else 1.0


def run_unit_tests():
    print("\n=== Task 2: Unit Tests ===")

    assert rle_decode_basic(rle_encode_basic(b'\xCF' * 5)) == b'\xCF' * 5
    print("  ok - stage 1 basic encode/decode")

    ex = bytes([0xCF, 0xCE, 0xCF, 0xCE, 0xCF])
    assert rle_encode_msb(ex)[0] == 0x85
    assert rle_decode_msb(rle_encode_msb(ex)) == ex
    assert rle_decode_msb(rle_encode_msb(b'\xCF' * 5)) == b'\xCF' * 5
    print("  ok - stage 2 MSB flag, 0x85 example matches assignment")

    for d, ms in [(b'\xAA' * 200, 1),
                  (bytes(range(256)) * 4, 1),
                  (bytes([0xCF, 0xCE]) * 3, 2),
                  (bytes([255, 0, 0]) * 10 + bytes([0, 255, 0]) * 5, 3)]:
        assert rle_decode_general(rle_encode_general(d, ms), ms) == d
    print("  ok - stage 3 Ms=1,2,3")

    text = "ААААББААААА"
    u32  = utf8_to_utf32(text.encode('utf-8'))
    dec  = rle_decode_general(rle_encode_general(u32, 4), 4)
    assert utf32_to_utf8(dec) == text.encode('utf-8')
    print("  ok - UTF-8 cyrillic roundtrip via UTF-32")

    with tempfile.TemporaryDirectory() as tmp:
        data = bytes(range(256)) * 50 + b'\xFF' * 1000
        open(f'{tmp}/orig.bin', 'wb').write(data)
        rle_encode_file(f'{tmp}/orig.bin', f'{tmp}/enc.rle')
        rle_decode_file(f'{tmp}/enc.rle',  f'{tmp}/dec.bin')
        assert open(f'{tmp}/dec.bin', 'rb').read() == data
    print("  ok - stage 4 file roundtrip")

    print("  all tests passed")
