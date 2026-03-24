import os
import struct
from PIL import Image, ImageDraw
import numpy as np

# file header magic bytes
MAGIC = b'RAWIMG'

# image type codes
BW    = 0
GRAY  = 1
COLOR = 2

def image_to_raw(img_path, raw_path, img_type):
    img = Image.open(img_path)
    w, h = img.size

    if img_type == BW:
        img = img.convert('1')
    elif img_type == GRAY:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    pixels = list(img.getdata())

    os.makedirs(os.path.dirname(raw_path) or '.', exist_ok=True)
    with open(raw_path, 'wb') as f:
        # write header: magic + type + width + height
        f.write(MAGIC)
        f.write(bytes([img_type]))
        f.write(struct.pack('>II', w, h))

        # write pixel data
        if img_type == BW:
            for p in pixels:
                f.write(b'\x01' if p else b'\x00')
        elif img_type == GRAY:
            f.write(bytes(pixels))
        else:
            for p in pixels:
                f.write(bytes(p[:3]))

    orig_size = os.path.getsize(img_path)
    raw_size  = os.path.getsize(raw_path)
    return orig_size, raw_size


def raw_to_image(raw_path, out_path):
    with open(raw_path, 'rb') as f:
        magic = f.read(6)
        assert magic == MAGIC
        t     = f.read(1)[0]
        w, h  = struct.unpack('>II', f.read(8))
        data  = f.read()

    if t == COLOR:
        img = Image.new('RGB', (w, h))
        pixels = [(data[i], data[i+1], data[i+2]) for i in range(0, len(data), 3)]
    else:
        img = Image.new('L', (w, h))
        if t == BW:
            pixels = [255 if b else 0 for b in data]
        else:
            pixels = list(data)

    img.putdata(pixels)
    img.save(out_path)


def generate_test_images(out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    W, H = 800, 600

    # black and white image
    bw  = Image.new('1', (W, H), 0)
    d   = ImageDraw.Draw(bw)
    for i in range(0, W, 40):
        d.rectangle([i, 0, i+20, H], fill=1)
    d.ellipse([200, 100, 600, 500], fill=1)
    bw.save(f'{out_dir}/test_bw.png')

    # grayscale image
    arr = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        arr[y, :] = int(y / H * 255)
    arr[100:200, 100:700] = 200
    arr[300:400, 100:700] = 80
    Image.fromarray(arr, 'L').save(f'{out_dir}/test_gray.png')

    # color image
    arr_c = np.zeros((H, W, 3), dtype=np.uint8)
    for x in range(W):
        arr_c[:, x, 0] = int(x / W * 255)
    for y in range(H):
        arr_c[y, :, 1] = int(y / H * 255)
    arr_c[250:350, :, 2] = 200
    Image.fromarray(arr_c, 'RGB').save(f'{out_dir}/test_color.png')


def run_task1(data_dir='data', out_dir='output'):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(f'{data_dir}/test_bw.png'):
        generate_test_images(data_dir)

    print("\n=== Task 1: Image RAW Format ===")
    print(f"{'File':<25} {'Original':>10} {'RAW':>12} {'Ratio':>8}")

    for name, t in [('test_bw', BW), ('test_gray', GRAY), ('test_color', COLOR)]:
        src  = f'{data_dir}/{name}.png'
        raw  = f'{out_dir}/{name}.raw'
        orig, raw_sz = image_to_raw(src, raw, t)
        print(f"{name+'.png':<25} {orig:>10,} {raw_sz:>12,} {raw_sz/orig:>7.1f}x")
        raw_to_image(raw, f'{out_dir}/{name}_restored.png')


if __name__ == '__main__':
    run_task1()
