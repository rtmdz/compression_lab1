import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from task6_lz import (lzss_encode, lzss_decode, lzw_encode, lzw_decode,
                       lz78_encode, lz78_decode, lz77_encode, lz77_decode,
                       run_unit_tests)

DATA = 'data'
OUT  = 'output'
os.makedirs(OUT, exist_ok=True)

def analyze_all(files):
    print(f"\n  {'File':<25} {'Algo':<6} {'Orig KB':>9} {'Enc KB':>9} {'Ratio':>7} {'Time':>7}  OK")
    print("  " + "-"*72)
    for path, label in files:
        if not os.path.exists(path): continue
        data = open(path, 'rb').read()
        for algo, enc_fn, dec_fn in [
            ('lzss', lzss_encode, lzss_decode),
            ('lz78', lz78_encode, lz78_decode),
            ('lzw',  lzw_encode,  lzw_decode),
        ]:
            t0      = time.time()
            enc     = enc_fn(data)
            elapsed = time.time() - t0
            dec     = dec_fn(enc)
            ok      = dec == data
            ratio   = len(data) / len(enc) if enc else 0
            print(f"  {label:<25} {algo:<6} {len(data)/1024:>9.1f} {len(enc)/1024:>9.1f} {ratio:>7.3f} {elapsed:>6.2f}s  {'ok' if ok else 'FAIL'}")

def plot_lzss_buffer(data, label):
    sizes  = [64, 128, 256, 512, 1024, 2048, 4096]
    ratios = []
    times  = []
    for buf in sizes:
        t0      = time.time()
        enc     = lzss_encode(data, buf)
        elapsed = time.time() - t0
        ratios.append(len(data) / len(enc))
        times.append(elapsed)
        print(f"  LZSS buffer={buf:5d}: ratio={ratios[-1]:.3f}, time={elapsed:.2f}s")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(sizes, ratios, marker='o', color='steelblue')
    ax1.set_xlabel('Buffer size'); ax1.set_ylabel('Compression ratio')
    ax1.set_title(f'LZSS ratio vs buffer ({label})'); ax1.grid(True)
    ax2.plot(sizes, times, marker='o', color='tomato')
    ax2.set_xlabel('Buffer size'); ax2.set_ylabel('Time (s)')
    ax2.set_title(f'LZSS time vs buffer ({label})'); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUT}/lzss_buffer_plot.png'); plt.close()
    print(f"  saved {OUT}/lzss_buffer_plot.png")

def plot_lzw_dict(data, label):
    sizes  = [256, 512, 1024, 2048, 4096, 8192, 16384]
    ratios = []
    times  = []
    for d in sizes:
        t0      = time.time()
        enc     = lzw_encode(data, d)
        elapsed = time.time() - t0
        ratios.append(len(data) / len(enc))
        times.append(elapsed)
        print(f"  LZW  dict={d:6d}: ratio={ratios[-1]:.3f}, time={elapsed:.2f}s")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(sizes, ratios, marker='o', color='green')
    ax1.set_xlabel('Dict size'); ax1.set_ylabel('Compression ratio')
    ax1.set_title(f'LZW ratio vs dict ({label})'); ax1.grid(True)
    ax2.plot(sizes, times, marker='o', color='orange')
    ax2.set_xlabel('Dict size'); ax2.set_ylabel('Time (s)')
    ax2.set_title(f'LZW time vs dict ({label})'); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUT}/lzw_dict_plot.png'); plt.close()
    print(f"  saved {OUT}/lzw_dict_plot.png")


if __name__ == '__main__':
    run_unit_tests()

    enwik   = f'{DATA}/enwik7.txt'
    russian = f'{DATA}/russian_text.txt'

    # 30KB slices — fast enough for pure python
    small_en = f'{OUT}/enwik7_30k.txt'
    small_ru = f'{OUT}/russian_30k.txt'
    if os.path.exists(enwik):
        open(small_en, 'wb').write(open(enwik,   'rb').read(30_000))
    if os.path.exists(russian):
        open(small_ru, 'wb').write(open(russian, 'rb').read(30_000))

    test_files = [
        (small_en,               'enwik7 (30KB)'),
        (small_ru,               'russian (30KB)'),
        (f'{OUT}/test_bw.raw',  'test_bw.raw'),
    ]

    print("\n=== All LZ algorithms ===")
    analyze_all(test_files)

    print("\n=== LZSS: buffer size analysis ===")
    if os.path.exists(small_en):
        plot_lzss_buffer(open(small_en,'rb').read(), 'enwik7 30KB')

    print("\n=== LZW: dict size analysis ===")
    if os.path.exists(small_en):
        plot_lzw_dict(open(small_en,'rb').read(), 'enwik7 30KB')

    print("\ndone! check output/ folder")
