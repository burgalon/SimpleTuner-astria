import gc
import os
import time

from pruna_pro import SmashConfig, smash
from pruna import PrunaModel
HOT_SWAP_SLOTS = 0
# IDENTITY_LORA = "nifleisch/flux-identity-lora"
# IDENTITY_LORA = "/data/models/1979152.safetensors"
IDENTITY_LORA = "/data/models/identity.safetensors"

def smash_pipe(orig_pipe, load_identity=False):
    if not os.environ.get('PRUNA'):
        return orig_pipe

    # smash_config = SmashConfig()
    # From predict.py
    # smash_config["compiler"] = "torch_compile"
    # smash_config["cacher"] = "auto"
    # smash_config["auto_objective"] = "quality"
    # smash_config["auto_cache_mode"] = "taylor"
    # smash_config._prepare_saving = False

    # From conversation with Pruna
    smash_config = SmashConfig()
    # smash_config["compiler"] = "torch_compile"
    smash_config["cacher"] = "auto"
    smash_config["auto_objective"] = "fidelity"
    smash_config["auto_cache_mode"] = "taylor"
    smash_config["auto_speed_factor"] = 0.6
    smash_config._prepare_saving = False
    start_time = time.time()
    pipe = smash(model=orig_pipe, smash_config=smash_config)
    end_time = time.time()
    print(f"Time taken for smash: {end_time - start_time:.2f} seconds")
    # pipe_to_call.cache_helper.enable() ??

    if load_identity:
        # TODO either before all per the warning, or after after the first call
        # pipe.enable_lora_hotswap(target_rank=128, check_compiled="warn")
        if HOT_SWAP_SLOTS>0:
            for i in range(HOT_SWAP_SLOTS):
                pipe.load_lora_weights(IDENTITY_LORA, adapter_name=f"default_{i}", low_cpu_mem_usage=False)
                # TODO not sure this is correct but seems to be the case in predict.py
                if i == 0:
                    pipe.enable_lora_hotswap(target_rank=128, check_compiled="warn")
            pipe.set_adapters([f"default_{i}" for i in range(HOT_SWAP_SLOTS)])

    return pipe

def scale_lora(pipe, adapter_name, factor):
    all_lora_keys = {
        k: v
        for k, v in pipe.transformer.state_dict().items()
        if ("lora_A" in k and adapter_name in k)
    }
    for tensor in all_lora_keys.values():
        tensor.mul_(factor)

def is_pruna_model(pipe):
    return hasattr(pipe, '_PrunaProModel__internal_model_ref')


if __name__ == "__main__":
    os.environ['PRUNA'] = '1'
    from diffusers import FluxPipeline
    import torch

    # print(f"Testing with HOT_SWAP_SLOTS={HOT_SWAP_SLOTS}")
    for i in range(5):
        generator = torch.Generator(device="cuda").manual_seed(i)

        orig_pipe = FluxPipeline.from_pretrained(
            '/data/models/1504944-flux1',
            torch_dtype=torch.bfloat16,
        ).to('cuda')
        start_time = time.time()
        orig_pipe('woman holding flowers', generator=generator).images[0].save(f'/data/models/smash_helper_test_{i}_orig.jpg')
        end_time = time.time()
        print(f"Orig {end_time - start_time:.2f}s")

        pipe = smash_pipe(orig_pipe, True)
        start_time = time.time()
        pipe('woman holding flowers', generator=generator).images[0].save(f'/data/models/smash_helper_test_{i}smashed.jpg')
        end_time = time.time()
        print(f"Smashed {end_time - start_time:.2f}s")
        pipe.destroy()
        del pipe
        gc.collect()



    # times = []
    # for i in range(5):
    #     start_time = time.time()
    #     pipe('woman holding flowers')
    #     end_time = time.time()
    #     if i>0:
    #         times.append(end_time - start_time)
    #         print(f"Time taken for first call: {end_time - start_time} seconds")
    # print(f"Average time taken for first call: {sum(times) / len(times)} seconds")
