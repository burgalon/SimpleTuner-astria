from diffusers import FluxPipeline
from peft import get_peft_model_state_dict
import torch
from diffusers.utils import (
    convert_state_dict_to_diffusers,
)

pipe = FluxPipeline.from_pretrained(
    '/data/models/1504944-flux1',
    torch_dtype=torch.bfloat16,
).to('cuda')

identity_lora = pipe.lora_state_dict("/data/models/1979152.safetensors")
for k in identity_lora:
    identity_lora[k] = identity_lora[k] * 0.0
pipe.load_lora_weights(identity_lora, adapter_name="default")
pipe.save_lora_weights(
    '/data/models/identity.safetensors',
    transformer_lora_layers=get_peft_model_state_dict(pipe.transformer),
    safe_serialization=True
)
pipe.unload_lora_weights()

# Sanity check that the identity lora is loaded correctly
pipe.load_lora_weights('/data/models/identity.safetensors', adapter_name="default")