import os
import time
from task7_final import (bwt_encode_file, bwt_decode_file,
                          canonical_huffman_encode_file, canonical_huffman_decode_file,
                          run_unit_tests)

DATA = 'data'
OUT  = 'output'
os.makedirs(OUT, exist_ok=True)

def analyze_bwt(files):
    print(f"\n  {'File':<25} {'Orig KB':>10} {'Enc KB':>10} {'Ratio':>8}  OK")
    print("  " + "-"*58)
    for path, label in files:
        if not os.path.exists(path): continue
        enc  = f'{OUT}/{os.path.basename(path)}.bwt3'
        dec  = f'{OUT}/{os.path.basename(path)}.bwt3.dec'
        orig, enc_sz = bwt_encode_file(path, enc)
        bwt_decode_file(enc, dec)
        ok   = open(dec,'rb').read() == open(path,'rb').read()
        ratio = orig / enc_sz if enc_sz else 0
        print(f"  {label:<25} {orig/1024:>10.1f} {enc_sz/1024:>10.1f} {ratio:>8.3f}  {'ok' if ok else 'FAIL'}")

def analyze_canonical_huffman(files):
    print(f"\n  {'File':<25} {'Orig KB':>10} {'Enc KB':>10} {'Ratio':>8}  OK")
    print("  " + "-"*58)
    for path, label in files:
        if not os.path.exists(path): continue
        enc  = f'{OUT}/{os.path.basename(path)}.chf'
        dec  = f'{OUT}/{os.path.basename(path)}.chf.dec'
        orig, enc_sz = canonical_huffman_encode_file(path, enc)
        canonical_huffman_decode_file(enc, dec)
        ok   = open(dec,'rb').read() == open(path,'rb').read()
        ratio = orig / enc_sz if enc_sz else 0
        print(f"  {label:<25} {orig/1024:>10.1f} {enc_sz/1024:>10.1f} {ratio:>8.3f}  {'ok' if ok else 'FAIL'}")


if __name__ == '__main__':
    run_unit_tests()

    enwik   = f'{DATA}/enwik7.txt'
    russian = f'{DATA}/russian_text.txt'

    small_en = f'{OUT}/enwik7_50k.txt'
    small_ru = f'{OUT}/russian_50k.txt'
    if os.path.exists(enwik):
        open(small_en, 'wb').write(open(enwik,   'rb').read(50_000))
    if os.path.exists(russian):
        open(small_ru, 'wb').write(open(russian, 'rb').read(50_000))

    files = [
        (small_en,               'enwik7 (50KB)'),
        (small_ru,               'russian (50KB)'),
        (f'{OUT}/test_bw.raw',  'test_bw.raw'),
        (f'{OUT}/test_gray.raw','test_gray.raw'),
    ]

    print("\n=== Efficient BWT (Suffix Array) + RLE ===")
    analyze_bwt(files)

    print("\n=== Canonical Huffman ===")
    analyze_canonical_huffman(files)

    print("\ndone! check output/ folder")
