CONTROLNETS_DICT = {
    'flux1': {
        'canny': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
        'depth': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
        'pose': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
        # 'pose': 'raulc0399/flux_dev_openpose_controlnet',
        'low_quality': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
        'tile': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
        'blur': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
        'gray': 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0',
    }
}

# https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0
RECOMMENDED_CONTROLNET_CONSTANTS = {
    'canny': {
        'controlnet_conditioning_scale': 0.7,
        'control_guidance_end': 0.8,
    },
    'softedge': {
        'controlnet_conditioning_scale': 0.7,
        'control_guidance_end': 0.8,
    },
    'depth': {
        'controlnet_conditioning_scale': 0.8,
        'control_guidance_end': 0.8,
    },
    'pose': {
        'controlnet_conditioning_scale': 0.9,
        'control_guidance_end': 0.65,
    },
    'gray': {
        'controlnet_conditioning_scale': 0.9,
        'control_guidance_end': 0.8,
    },
}
