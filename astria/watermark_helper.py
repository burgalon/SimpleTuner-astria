import PIL
from PIL import Image
from image_utils import load_image

def add_watermark(images, watermark):
    # Percentage constants
    padding = 2
    watermark_x = 100
    watermark_y = 100
    watermark_size = 10
    print("Adding watermark", watermark)
    watermark = load_image(watermark, 'RGBA')

    for i, img in enumerate(images):
        # Get image dimensions
        img_width, img_height = img.size

        # Calculate watermark size based on image dimensions and user input
        k = (float(img_width * img_height) / (watermark.width * watermark.height)) ** 0.5
        watermark_width = int(watermark.width * k * watermark_size / 100)
        watermark_height = int(watermark.height * k * watermark_size / 100)

        # Calculate adjusted position with padding
        adjusted_x = int(max(0, min(img_width - watermark_width - (padding * img_width / 100), watermark_x * img_width / 100)))
        adjusted_y = int(max(0, min(img_height - watermark_height - (padding * img_height / 100), watermark_y * img_height / 100)))

        # Resize watermark if needed
        if watermark_width != watermark.width or watermark_height != watermark.height:
            watermark = watermark.resize((watermark_width, watermark_height), PIL.Image.Resampling.LANCZOS)


        # Create a new image with alpha channel for watermark placement
        composite_img = Image.new("RGBA", img.size)

        # Paste the original image and watermark
        composite_img.paste(img, (0, 0))
        composite_img.paste(watermark, (adjusted_x, adjusted_y), watermark)

        # Update processed image with watermarked version (replace with OpenCV code if using OpenCV)
        images[i] = composite_img.convert("RGB")
    return images
