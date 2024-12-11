CONTROLNETS_DICT = {
    'flux1': {
        'canny': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
        'depth': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
        'pose': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
        # 'pose': 'raulc0399/flux_dev_openpose_controlnet',
        'low_quality': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
        'tile': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
        'blur': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
        'gray': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
    }
}

# https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro#model-cards
CONTROL_MODES = {
    'canny': 0,
    'tile': 1,
    'depth': 2,
    'blur': 3,
    'pose': 4,
    'gray': 5,
    'low_quality': 6,
}


