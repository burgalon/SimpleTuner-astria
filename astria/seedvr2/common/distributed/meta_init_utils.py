# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import torch
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened

__all__ = ["meta_non_persistent_buffer_init_fn"]


# def meta_non_persistent_buffer_init_fn(module: nn.Module) -> nn.Module:
#     """
#     Used for materializing `non-persistent tensor buffers` while model resuming.

#     Since non-persistent tensor buffers are not saved in state_dict,
#     when initializing model with meta device, user should materialize those buffers manually.

#     Currently, only `rope.dummy` is this special case.
#     """
#     with torch.no_grad():
#         for submodule in module.modules():
#             if not isinstance(submodule, RotaryEmbedding):
#                 continue
#             for buffer_name, buffer in submodule.named_buffers(recurse=False):
#                 if buffer.is_meta and "dummy" in buffer_name:
#                     materialized_buffer = torch.zeros_like(buffer, device="cpu")
#                     setattr(submodule, buffer_name, materialized_buffer)
#     assert not any(b.is_meta for n, b in module.named_buffers())
#     return module


def materialize_all_meta_buffers(module: nn.Module, device: str = "cpu") -> nn.Module:
    """
    Materialize ANY meta buffers found in the module tree onto `device`.
    Keeps the persistent flag of the original buffers.
    """
    with torch.no_grad():
        for sub in module.modules():
            # named_buffers(recurse=False) -> only this submodule's buffers
            for name, buf in sub.named_buffers(recurse=False):
                if getattr(buf, "is_meta", False):
                    # Try to preserve persistence flag if present
                    persistent = getattr(buf, "_is_persistent", True)
                    new = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
                    # replace via register_buffer to keep proper state
                    sub.register_buffer(name, new, persistent=persistent)
    return module

# Backwards-compatible alias if other code imports the old name
def meta_non_persistent_buffer_init_fn(module: nn.Module) -> nn.Module:
    # Now just call the robust version
    return materialize_all_meta_buffers(module, device="cpu")