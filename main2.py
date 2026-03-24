import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from task3_entropy import (calc_entropy, filter_ascii, mtf_encode, mtf_decode,
                            huffman_encode_file, huffman_decode_file,
                            arithmetic_encode, find_precision_limit,
                            run_unit_tests as run_entropy_tests)
from task4_bwt    import (bwt_encode_file, bwt_decode_file,
                           run_unit_tests as run_bwt_tests)

DATA = 'data'
OUT  = 'output'
os.makedirs(OUT, exist_ok=True)

def plot_entropy(data_path):
    data    = filter_ascii(open(data_path, 'rb').read())
    ms_vals = [1, 2, 3, 4]
    ents    = []
    for ms in ms_vals:
        trimmed = data[:len(data) - len(data) % ms]
        e       = calc_entropy(trimmed, Ms=ms)
        ents.append(e)
        print(f"  Ms={ms}: entropy = {e:.4f} bits/symbol")
    plt.figure(figsize=(7, 4))
    plt.plot(ms_vals, ents, marker='o', color='steelblue')
    plt.xlabel('Symbol size (bytes)')
    plt.ylabel('Entropy (bits/symbol)')
    plt.title('Entropy vs symbol size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUT}/entropy_plot.png')
    plt.close()
    print(f"  plot saved to {OUT}/entropy_plot.png")

def analyze_mtf(data_path):
    data     = open(data_path, 'rb').read()[:50000]
    e_before = calc_entropy(data, Ms=1)
    mtf_data = mtf_encode(data)
    e_after  = calc_entropy(mtf_data, Ms=1)
    print(f"  entropy before MTF: {e_before:.4f}")
    print(f"  entropy after  MTF: {e_after:.4f}")
    direction = 'reduced' if e_after < e_before else 'increased'
    print(f"  MTF {direction} entropy by {abs(e_after-e_before):.4f} bits/symbol")
    assert mtf_decode(mtf_encode(data)) == data
    print("  MTF roundtrip ok")

def analyze_huffman(files):
    print(f"\n  {'File':<30} {'Orig KB':>10} {'Enc KB':>10} {'Ratio':>8}")
    print("  " + "-"*60)
    for path, label in files:
        if not os.path.exists(path): continue
        enc  = f'{OUT}/{os.path.basename(path)}.huf'
        dec  = f'{OUT}/{os.path.basename(path)}.huf.dec'
        orig, enc_sz = huffman_encode_file(path, enc)
        huffman_decode_file(enc, dec)
        ok   = open(dec, 'rb').read() == open(path, 'rb').read()
        ratio = orig / enc_sz if enc_sz else 0
        print(f"  {label:<30} {orig/1024:>10.1f} {enc_sz/1024:>10.1f} {ratio:>8.3f}  {'ok' if ok else 'FAIL'}")

def analyze_arithmetic():
    limit = find_precision_limit()
    print(f"  precision collapses at string length ~{limit}")
    print("  (low >= high in float64 after this point)")

def analyze_bwt(files):
    print(f"\n  {'File':<30} {'Orig KB':>10} {'Enc KB':>10} {'Ratio':>8}")
    print("  " + "-"*60)
    for path, label in files:
        if not os.path.exists(path): continue
        enc  = f'{OUT}/{os.path.basename(path)}.bwt'
        dec  = f'{OUT}/{os.path.basename(path)}.bwt.dec'
        orig, enc_sz = bwt_encode_file(path, enc)
        bwt_decode_file(enc, dec)
        ok   = open(dec, 'rb').read() == open(path, 'rb').read()
        ratio = orig / enc_sz if enc_sz else 0
        print(f"  {label:<30} {orig/1024:>10.1f} {enc_sz/1024:>10.1f} {ratio:>8.3f}  {'ok' if ok else 'FAIL'}")


if __name__ == '__main__':
    run_entropy_tests()
    run_bwt_tests()

    enwik   = f'{DATA}/enwik7.txt'
    russian = f'{DATA}/russian_text.txt'

    # use small slice of enwik for huffman (full file is slow due to pure python)
    small_enwik = f'{OUT}/enwik7_small.txt'
    if os.path.exists(enwik):
        open(small_enwik, 'wb').write(open(enwik, 'rb').read(200_000))

    huffman_files = [
        (small_enwik,               'enwik7 (200KB)'),
        (russian,                   'russian_text.txt'),
        (f'{OUT}/test_bw.raw',     'test_bw.raw'),
        (f'{OUT}/test_gray.raw',   'test_gray.raw'),
    ]

    bwt_files = [
        (small_enwik,  'enwik7 (200KB)'),
        (russian,      'russian_text.txt'),
    ]

    print("\n=== Entropy vs symbol size ===")
    if os.path.exists(enwik): plot_entropy(enwik)

    print("\n=== MTF effect on entropy ===")
    if os.path.exists(enwik): analyze_mtf(enwik)

    print("\n=== Huffman compression ===")
    analyze_huffman(huffman_files)

    print("\n=== Arithmetic coding precision ===")
    analyze_arithmetic()

    print("\n=== BWT + RLE compression ===")
    analyze_bwt(bwt_files)

    print("\ndone! check output/ folder")
