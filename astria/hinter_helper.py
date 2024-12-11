from PIL import Image, ImageEnhance, ImageOps
from controlnet_aux import OpenposeDetector, CannyDetector
from transformers import pipeline

device = 'cuda'
annotator_ckpts_path = "/data/cache/annotator_ckpts"
#         filename = filename or "annotator/ckpts/network-bsds500.pth"
#         filename = filename or "network-bsds500.pth"
#     filename = filename or "sk_model.pth"
#     coarse_filename = coarse_filename or "sk_model2.pth"
#     filename = filename or "netG.pth"
#         filename = filename or "annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
#         filename = filename or "dpt_hybrid-midas-501f0c75.pt"
# "dpt_large": os.path.join(annotator_ckpts_path, "dpt_large-midas-2f21e586.pt"),
# "dpt_hybrid": os.path.join(annotator_ckpts_path, "dpt_hybrid-midas-501f0c75.pt"),
# te_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
#         filename = filename or "annotator/ckpts/mlsd_large_512_fp32.pth"
#         filename = filename or "mlsd_large_512_fp32.pth"
#     pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
#     pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
#     filename = filename or "scannet.pt"

def high_contrast_black_white(img: Image) -> Image:
    # Convert to grayscale (black and white)
    gray = img.convert("L")

    # Check if the image is mostly white
    avg_pixel_value = sum(gray.getdata()) / (gray.width * gray.height)
    is_mostly_white = avg_pixel_value > 200

    # Adjust levels to make the image high contrast
    min_val, max_val = gray.getextrema()
    # Setting the threshold at the middle of min_val and max_val
    threshold = (min_val + max_val) / 2

    # Binarize the image: pixels below the threshold will be black, above will be white.
    binarized = gray.point(lambda p: 255 if p > threshold else 0)

    # Invert if the image is mostly white
    if is_mostly_white:
        binarized = ImageOps.invert(binarized)

    return binarized.convert("RGB")

class DepthAnythingV2Wrapper:
    def __init__(self):
        self.detector = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    def __call__(self, img: Image) -> Image:
        return self.detector(img)["depth"].convert('RGB').resize(img.size)

CONTROL_TYPES_TO_HINTER = {
    "depthanythingv2": lambda: DepthAnythingV2Wrapper(),
    # "hed": lambda: HEDdetector.from_pretrained(annotator_ckpts_path).to(device),
    # "midas": lambda: MidasDetector.from_pretrained(annotator_ckpts_path).to(device),
    # "mlsd": lambda: MLSDdetector.from_pretrained(annotator_ckpts_path).to(device),
    # det_ckpt: https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
    # pose_ckpt: https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth
    # "pose": lambda: DWposeDetector(det_config='./examples/dreambooth/dwpose_helper/yolox_l_8xb8-300e_coco.py', pose_config='./examples/dreambooth/dwpose_helper/dwpose-l_384x288.py', det_ckpt='/data/cache/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth', pose_ckpt='/data/cache/dw-ll_ucoco_384.pth', device=device), # OpenposeDetector.from_pretrained(annotator_ckpts_path).to(device),
    "pose": lambda: OpenposeDetector.from_pretrained(annotator_ckpts_path, filename='body_pose_model.pth').to(device),
    # for scribble or softedge
    # "pidi": lambda: PidiNetDetector.from_pretrained(annotator_ckpts_path).to(device),
    # "normal_bae": lambda: NormalBaeDetector.from_pretrained(annotator_ckpts_path).to(device),
    # "lineart": lambda: LineartDetector.from_pretrained(annotator_ckpts_path).to(device),
    # "lineart_anime": lambda: LineartAnimeDetector.from_pretrained(PATH).to(device),
    # alt depth
    # "zoe": lambda: ZoeDetector.from_pretrained(PATH).to(device),
    # "depth_leres": lambda: LeresDetector.from_pretrained(annotator_ckpts_path, boost=False).to(device),
    # "depth_leres++": lambda: LeresDetector.from_pretrained(annotator_ckpts_path, boost=True).to(device),
    # "seg": lambda: SamDetector.from_pretrained(annotator_ckpts_path),
    "canny": lambda: CannyDetector(),
    # "shuffle": lambda: ContentShuffleDetector().to(device),
    # "mediapipe_face": lambda: MediapipeFaceDetector(),
    # "ip2p": lambda: lambda img: img,
    "tile": lambda: lambda img: img,
    "blur": lambda: lambda img: img,
    "low_quality": lambda: lambda img: img,
    # "qr": lambda: lambda img: high_contrast_black_white(img),
    # inpaint: optical flow or mask
    # "inpaint": lambda: lambda img: img,
    # the image itself - should be resized to the same size as input image
}

CONTROLNET_TO_HINTER = {
    "depth": "depthanythingv2", # or zoe
    "softedge": "pidi",
    "scribble": "pidi", # or pidi
    "normal": "normal_bae",
    "pose_with_face": "pose",
    "pose_with_hand": "pose",
    "pose_face_and_hand": "pose",
}


def get_detector(control_type):
    hinter_type = CONTROLNET_TO_HINTER.get(control_type, control_type)
    return CONTROL_TYPES_TO_HINTER[hinter_type]()
