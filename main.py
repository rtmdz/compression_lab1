import os
import sys
import random
import urllib.request
import zipfile

from task1_images import run_task1
from task2_rle    import rle_encode_file, rle_decode_file, estimate_rle_ratio, run_unit_tests

DATA = 'data'
OUT  = 'output'
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT,  exist_ok=True)


def prep_enwik7():
    path = f'{DATA}/enwik7.txt'
    if os.path.exists(path) and os.path.getsize(path) == 10_000_000:
        return path
    try:
        print("  downloading enwik8...")
        urllib.request.urlretrieve('https://mattmahoney.net/dc/enwik8.zip', f'{DATA}/enwik8.zip')
        with zipfile.ZipFile(f'{DATA}/enwik8.zip') as z:
            with z.open('enwik8') as src, open(path, 'wb') as dst:
                dst.write(src.read(10_000_000))
        os.remove(f'{DATA}/enwik8.zip')
    except:
        print("  download failed, generating synthetic text instead")
        words = "the of and to in is was he for it with as his on be at by this had not are but from or an they which one you were her all she there would their we him been has when who will more no if out so said".split()
        rng   = random.Random(42)
        text  = " ".join(rng.choice(words) for _ in range(1_500_000)) + "\n"
        open(path, 'w').write(text[:10_000_000])
    return path


def prep_russian():
    path = f'{DATA}/russian_text.txt'
    if os.path.exists(path) and os.path.getsize(path) >= 200_000:
        return path
    words = "привет мир текст данные сжатие алгоритм байт символ кодирование файл строка длина функция программа компьютер информация метод значение результат".split()
    rng   = random.Random(7)
    lines = []
    total = 0
    while total < 250_000:
        line  = " ".join(rng.choice(words) for _ in range(rng.randint(6, 18))) + ".\n"
        lines.append(line)
        total += len(line.encode('utf-8'))
    open(path, 'w', encoding='utf-8').writelines(lines)
    return path


def prep_binary():
    if os.path.exists(sys.executable) and os.path.getsize(sys.executable) >= 1_000_000:
        return sys.executable
    path = f'{DATA}/random.bin'
    if not os.path.exists(path):
        rng = random.Random(99)
        open(path, 'wb').write(bytes(rng.getrandbits(8) for _ in range(2_000_000)))
    return path


def analyze(path, label, Ms=1, Mc=1):
    data    = open(path, 'rb').read()
    est     = estimate_rle_ratio(data, Ms)
    enc     = f'{OUT}/{os.path.basename(path)}.rle'
    dec     = f'{OUT}/{os.path.basename(path)}.dec'
    orig, enc_sz = rle_encode_file(path, enc, Ms, Mc)
    rle_decode_file(enc, dec)
    ok      = open(dec, 'rb').read() == data
    actual  = orig / enc_sz if enc_sz else 0
    return dict(label=label, orig=orig, enc=enc_sz, est=est, actual=actual, ok=ok)


def print_report(results):
    print("\n=== Compression Ratio Report ===")
    print(f"{'File':<30} {'Orig KB':>10} {'Enc KB':>10} {'Estimated':>10} {'Actual':>8}  OK")
    print("-" * 74)
    for r in results:
        status = "yes" if r['ok'] else "FAIL"
        print(f"{r['label']:<30} {r['orig']/1024:>10.1f} {r['enc']/1024:>10.1f} "
              f"{r['est']:>10.3f} {r['actual']:>8.3f}  {status}")
    print()
    print("ratio < 1.0 means the file got bigger after compression")
    print("ratio > 1.0 means the file got smaller")


if __name__ == '__main__':
    run_unit_tests()
    run_task1(DATA, OUT)

    print("\n=== Preparing test data ===")
    enwik   = prep_enwik7()
    russian = prep_russian()
    binary  = prep_binary()
    print(f"  enwik7:  {os.path.getsize(enwik):,} bytes")
    print(f"  russian: {os.path.getsize(russian):,} bytes")
    print(f"  binary:  {os.path.getsize(binary):,} bytes")

    print("\n=== Running RLE on all files ===")
    results = [
        analyze(enwik,                    'enwik7.txt',           Ms=1),
        analyze(russian,                  'russian_text.txt',     Ms=1),
        analyze(binary,                   os.path.basename(binary), Ms=1),
        analyze(f'{OUT}/test_bw.raw',    'test_bw.raw',          Ms=1),
        analyze(f'{OUT}/test_gray.raw',  'test_gray.raw',        Ms=1),
        analyze(f'{OUT}/test_color.raw', 'test_color.raw',       Ms=3),
    ]
    print_report(results)
    print("done, check the output/ folder")
