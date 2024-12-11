from PIL import Image
import math
import numpy as np
from astria_utils import CACHE_DIR

def apply_hald_clut(hald_img, img):
    hald_w, hald_h = hald_img.size
    clut_size = int(round(math.pow(hald_w, 1/3)))
    # We square the clut_size because a 12-bit HaldCLUT has the same amount of information as a 144-bit 3D CLUT
    scale = (clut_size * clut_size - 1) / 255
    # Convert the PIL image to numpy array
    img = np.asarray(img)
    # We are reshaping to (144 * 144 * 144, 3) - it helps with indexing
    hald_img = np.asarray(hald_img).reshape(clut_size ** 6, 3)
    # Figure out the 3D CLUT indexes corresponding to the pixels in our image
    clut_r = np.rint(img[:, :, 0] * scale).astype(int)
    clut_g = np.rint(img[:, :, 1] * scale).astype(int)
    clut_b = np.rint(img[:, :, 2] * scale).astype(int)
    filtered_image = np.zeros((img.shape))
    # Convert the 3D CLUT indexes into indexes for our HaldCLUT numpy array and copy over the colors to the new image
    filtered_image[:, :] = hald_img[clut_r + clut_size ** 2 * clut_g + clut_size ** 4 * clut_b]
    filtered_image = Image.fromarray(filtered_image.astype('uint8'), 'RGB')
    return filtered_image


CLUT_DICT = {
    'Film Velvia': 'HaldCLUT/Color/Fuji/Fuji Velvia 50.png',
    'Film Portra': 'HaldCLUT/Color/Kodak/Kodak Portra 400 VC 4 ++.png',
    'Ektar': 'HaldCLUT/Color/Kodak/Kodak Ektar 100.png',
}

def add_clut(img: Image, clut: str):
    clut_filename = CLUT_DICT[clut]
    velvia_hald_clut = Image.open(f"{CACHE_DIR}/{clut_filename}")
    img = apply_hald_clut(velvia_hald_clut, img)
    return img

if __name__ == "__main__":
    import sys
    orig_image = Image.open(sys.argv[1])
    for key in CLUT_DICT.keys():
        image = add_clut(orig_image, key)
        image.save(sys.argv[1] + f".{key}.png")
