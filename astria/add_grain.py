from PIL import Image, ImageFilter
import numpy as np
import cv2

def add_grain(image: Image, val=0.010) -> Image:
    im_arr = np.array(image) / 255.0

    cols, rows = image.size

    noise_im = np.zeros((rows, cols))
    factor = 1
    while rows//factor > 256 and cols//factor > 256 and factor <= 2:
        print(f"factor={factor}")
        # noise_factor = generate_gaussian_noise((rows//factor, cols//factor), var=(val*factor)**2)
        # use numpy instead
        noise_factor = np.random.normal(0, val*factor, (rows//factor, cols//factor))
        # noise_im += resize(noise_factor, (rows, cols))
        noise_im += cv2.resize(noise_factor, (cols, rows))
        factor *= 2


    if len(im_arr.shape) > 2:
        noise_im =  np.stack([noise_im]*3, axis=2)  # Add the noise to the image
    noisy_img = im_arr + noise_im

    noisy_img = np.round((255 * noisy_img)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

if __name__ == "__main__":
    import sys
    # Load an image into a PIL Image object
    image = Image.open(sys.argv[1])
    # image = Image.fromarray(np.full((640, 512), 0.5)*255)
    # image = Image.fromarray(np.full((640*2*2, 512*2*2), 0.5)*255)

    # Generate the grainy image
    for val in [0.005, 0.008, 0.010, 0.020, 0.030]:
        grainy_image = add_grain(image, val)
        grainy_image.save(sys.argv[1]+f".grainy-{val}.jpg")
