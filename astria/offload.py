import os
import gc
import sys
import time
import shutil
import tempfile
import inspect
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

import torch
from torch.nn import Module

# Optional cache dir hint
try:
    from astria_utils import CACHE_DIR as _ASTRIA_CACHE_DIR
except Exception:
    _ASTRIA_CACHE_DIR = None

# Diffusers is optional at import time
try:
    from diffusers import DiffusionPipeline
except Exception:
    DiffusionPipeline = tuple()

# Optional Accelerate hooks cleanup (harmless if missing)
try:
    from accelerate.hooks import remove_hook_from_submodules
except Exception:
    def remove_hook_from_submodules(*args, **kwargs):
        return

# Tensorizer preferred
try:
    from tensorizer import TensorSerializer, TensorDeserializer
    _HAS_TENSORIZER = True
except Exception:
    _HAS_TENSORIZER = False

# Secondary: safetensors
try:
    from safetensors.torch import save_file as st_save_file, load_file as st_load_file
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False


# ------------------------- Simple CPU offload -------------------------

def _module_has_cuda_params(m):
    try:
        return any(p.is_cuda for p in m.parameters())
    except Exception:
        return False


@contextmanager
def offload_to_cpu(*mods):
    moved = []
    for m in mods:
        if m is None:
            continue
        try:
            m.to("cpu")
            moved.append(m)
        except Exception:
            for attr in ("model", "clip_vision_model"):
                try:
                    sub = getattr(m, attr, None)
                    if sub is not None:
                        sub.to("cpu")
                        moved.append(sub)
                except Exception:
                    pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        for m in moved:
            try:
                m.to("cuda")
            except Exception:
                pass


# ------------------------- Debug helpers -------------------------

def _enable_cuda_history():
    try:
        torch.cuda.memory._record_memory_history(enabled=True, record_context=True, trace_allocations=True)
        return True
    except TypeError:
        try:
            torch.cuda.memory._record_memory_history(True, True)
            return True
        except Exception:
            return False


def explain_vram(self, include_frames: bool = True, extra_roots: dict[str, object] | None = None, top: int = 40):
    dev_index = torch.cuda.current_device()

    from collections import defaultdict
    import types

    def _skip(o):
        t = type(o)
        m = t.__module__
        if o is None: return True
        if isinstance(o, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
                          types.MethodType, types.BuiltinMethodType, types.CodeType)):
            return True
        if m and m.startswith("torch._classes"):
            return True
        if isinstance(o, (str, bytes, int, float, bool)):
            return True
        return False

    ptr2owners, ptr2nbytes = defaultdict(list), {}
    visited = set()

    def _record_tensor(t, owner_path: str):
        if not (torch.is_tensor(t) and getattr(t, "is_cuda", False) and t.device.index == dev_index):
            return
        try:
            ptr = t.data_ptr()
            nbytes = t.nelement() * t.element_size()
            ptr2owners[ptr].append(owner_path)
            if ptr not in ptr2nbytes or nbytes > ptr2nbytes[ptr]:
                ptr2nbytes[ptr] = nbytes
        except Exception:
            pass

    def _walk(obj, path: str):
        if _skip(obj): return
        oid = id(obj)
        if oid in visited: return
        visited.add(oid)

        if torch.is_tensor(obj): _record_tensor(obj, path); return
        if isinstance(obj, torch.nn.Parameter): _record_tensor(obj, path); return

        if isinstance(obj, torch.nn.Module):
            try:
                for n, p in obj.named_parameters(recurse=True):
                    _record_tensor(p, f"{path}.{n}")
            except Exception: pass
            try:
                for n, b in obj.named_buffers(recurse=True):
                    if getattr(b, "is_cuda", False):
                        _record_tensor(b, f"{path}.{n}")
            except Exception: pass

        try:
            from diffusers import DiffusionPipeline as _DP
            if isinstance(obj, _DP):
                comps = getattr(obj, "components", None)
                if isinstance(comps, dict):
                    for k, v in list(comps.items()):
                        _walk(v, f"{path}.components[{repr(k)}]")
        except Exception:
            pass

        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if not _skip(v): _walk(v, f"{path}[{repr(k)}]")
            return

        if isinstance(obj, (list, tuple, set)):
            for i, v in enumerate(list(obj)):
                if not _skip(v): _walk(v, f"{path}[{i}]")
            return

        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for k, v in list(d.items()):
                if isinstance(k, str) and k.startswith("__") and k.endswith("__"): continue
                if not _skip(v): _walk(v, f"{path}.{k}")

    roots = {
        "self": self,
        "self.pipe": getattr(self, "pipe", None),
        "self.img2img": getattr(self, "img2img", None),
        "self.inpaint": getattr(self, "inpaint", None),
        "self.fill": getattr(self, "fill", None),
        "self.controlnet_txt2img": getattr(self, "controlnet_txt2img", None),
        "self.controlnet_img2img": getattr(self, "controlnet_img2img", None),
        "self.controlnet_inpaint_txt2img": getattr(self, "controlnet_inpaint_txt2img", None),
        "self.rag_diffusion_pipe": getattr(self, "rag_diffusion_pipe", None),
        "self.pulid_pipe": getattr(self, "pulid_pipe", None),
        "self.pulid_model": getattr(self, "pulid_model", None),
        "self.sr_model": getattr(self, "sr_model", None),
        "self.last_pipe": getattr(self, "last_pipe", None),
    }
    if extra_roots: roots.update(extra_roots)

    if include_frames:
        for f in gc.get_objects():
            import types as _types
            if isinstance(f, _types.FrameType):
                tag = f"[frame:{f.f_globals.get('__name__','?')}.{f.f_code.co_name}:{f.f_lineno}]"
                roots[tag + ".locals"]  = f.f_locals
                roots[tag + ".globals"] = f.f_globals

    for name, root in roots.items():
        if root is not None and not _skip(root):
            _walk(root, name)

    def _fmt(n):
        for u in ("B","KiB","MiB","GiB","TiB"):
            if n < 1024 or u == "TiB": return f"{n:.1f} {u}"
            n /= 1024.0

    live_bytes = torch.cuda.memory_allocated(dev_index)
    owner_bytes = {}
    known_ptr_bytes = 0
    for ptr, nbytes in ptr2nbytes.items():
        owners = ptr2owners.get(ptr) or []
        owner = owners[0] if owners else f"<ptr 0x{ptr:x}>"
        owner_bytes[owner] = owner_bytes.get(owner, 0) + nbytes
        known_ptr_bytes += nbytes

    active_states = {"active", "active_allocated", "allocated", "active_reuse"}
    allocs = []
    ptr_field = "addr"
    snap = None
    try:
        snap = torch.cuda.memory._snapshot()
        allocs = snap.get("device_allocations") or snap.get("segments") or []
        if allocs:
            ptr_field = "addr" if "addr" in allocs[0] else ("address" if "address" in allocs[0] else None)
            allocs = [a for a in allocs if a.get("state") in active_states]
    except Exception:
        pass

    unknown_bytes = max(0, live_bytes - known_ptr_bytes)

    print("\n========== VRAM ATTRIBUTION (by owner path) ==========")
    print(f"device={dev_index}  live_allocated={_fmt(live_bytes)}  "
          f"attributed(Python tensors)={_fmt(known_ptr_bytes)}  "
          f"unattributed(allocator/C++)={_fmt(unknown_bytes)}")
    print("-------------------------------------------------------")
    for path, b in sorted(owner_bytes.items(), key=lambda kv: kv[1], reverse=True)[:top]:
        print(f"{_fmt(b):>10}  {path}")

    if include_frames:
        print("\n(note) Paths starting with [frame:...] are locals/globals holding CUDA tensors.)")

    if snap and ptr_field:
        def _hint(a):
            frames = a.get("frames") or []
            if not frames: return ""
            fr = frames[-1]
            fn = fr.get("filename","?").rsplit("/", 2)[-1]
            return f" @ {fn}:{fr.get('line','?')} {fr.get('name','')}"
        unknown = []
        for a in allocs:
            try:
                if a.get(ptr_field) not in ptr2nbytes:
                    unknown.append(a)
            except Exception:
                pass
        unknown.sort(key=lambda a: a.get("size", 0), reverse=True)
        print("\n========== LARGEST UNATTRIBUTED ALLOCATIONS ==========")
        for a in unknown[:20]:
            sz = a.get("size", 0)
            seg = a.get("segment_type", "?")
            print(f"{_fmt(sz):>10}  seg={seg}{_hint(a)}")

    print('GPU Live Processes', torch.cuda.list_gpu_processes())
    print("currently allocated=", torch.cuda.memory_allocated() / 1024**3, "GiB",
          "currently reserved=", torch.cuda.memory_reserved() / 1024**3, "GiB")


def _atomic_path(final_path: str) -> str:
    return f"{final_path}.tmp-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex}"

def _serialize_module_tensorizer(mod: Module, path: str):
    tmp = _atomic_path(path)
    with open(tmp, "wb") as f:
        s = TensorSerializer(f)
        s.write_module(mod, include_non_persistent_buffers=False)
        s.close()
    os.replace(tmp, path)

def _serialize_module_safetensors(mod: Module, path: str):
    tmp = _atomic_path(path)
    sd = {}
    for k, v in mod.state_dict().items():
        t = v.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        # keep GPU tensors on GPU here; safetensors will stage as needed
        sd[k] = t
    st_save_file(sd, tmp, metadata={"astria": "offload"})
    os.replace(tmp, path)

def _serialize_module_torchsave(mod: Module, path: str):
    tmp = _atomic_path(path)
    sd = mod.state_dict()
    torch.save(sd, tmp, _use_new_zipfile_serialization=True)
    os.replace(tmp, path)

def _device_and_index(dev: torch.device | str | int | None):
    if isinstance(dev, torch.device):
        d = dev
    elif isinstance(dev, int):
        d = torch.device(f"cuda:{dev}")
    elif isinstance(dev, str):
        d = torch.device(dev)
    else:
        d = torch.device("cuda", torch.cuda.current_device())
    if d.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {d}")
    idx = d.index if d.index is not None else torch.cuda.current_device()
    return d, idx


def report_cuda_memory(dev: torch.device | str | int):
    d, idx = _device_and_index(dev)
    with torch.cuda.device(d):
        try:
            free, total = torch.cuda.mem_get_info()
            used = total - free
        except AttributeError:
            props = torch.cuda.get_device_properties(idx)
            total = props.total_memory
            used = torch.cuda.memory_reserved(idx)
            free = total - used
    allocated = torch.cuda.memory_allocated(idx)
    reserved  = torch.cuda.memory_reserved(idx)
    gib = 1024 ** 3
    print(f"[device {d}] device-wide used={used/gib:.2f} GiB, free={free/gib:.2f} GiB / total={total/gib:.2f} GiB")
    print(f"[process]  allocated={allocated/gib:.2f} GiB (live), reserved={reserved/gib:.2f} GiB (cache), "
          f"headroom≈{(total-reserved)/gib:.2f} GiB")


def _tensor_storage_nbytes(t: torch.Tensor) -> int:
    try:
        st = t.untyped_storage()
        if hasattr(st, "is_cuda") and not st.is_cuda:
            return 0
        if hasattr(st, "device") and getattr(st.device(), "type", None) != "cuda":
            return 0
        if hasattr(st, "nbytes") and st.nbytes() == 0:
            return 0
        if hasattr(st, "data_ptr") and st.data_ptr() == 0:
            return 0
        return int(st.nbytes()) if hasattr(st, "nbytes") else int(st.size() * st.dtype.itemsize)
    except Exception:
        return 0


def list_live_cuda_tensors(limit: int = 50):
    seen = {}
    for obj in gc.get_objects():
        try:
            t = None
            if torch.is_tensor(obj) and getattr(obj, "is_cuda", False):
                t = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data) and getattr(obj.data, "is_cuda", False):
                t = obj.data
            if t is None:
                continue
            nbytes = _tensor_storage_nbytes(t)
            if nbytes > 0:
                seen[id(t)] = (t, nbytes)
        except Exception:
            pass

    ts = sorted(seen.values(), key=lambda kv: kv[1], reverse=True)
    total_bytes = sum(n for _, n in ts)
    print(f"Live CUDA tensors (with real storage): {len(ts)}, total ~{total_bytes/1024**3:.2f} GiB")
    for t, nbytes in ts[:limit]:
        shape_str = str(tuple(t.shape))
        print(f"  {shape_str:>24} {str(t.dtype):>10} {str(t.device):>8}  ~{nbytes/1024**2:8.1f} MiB")
    return [t for t,_ in ts]


def _bytes_on_module_params(module: Module) -> int:
    b = 0
    try:
        for p in module.parameters(recurse=True):
            b += p.nelement() * p.element_size()
        for bfr in module.buffers(recurse=True):
            b += bfr.nelement() * bfr.element_size()
    except Exception:
        pass
    return b


def summarize_objects_cuda(objs, only_nonzero: bool = True):
    def _fmt_mib(nbytes: int) -> str:
        return f"{nbytes/1024**2:8.1f} MiB"

    for obj in objs:
        if obj is None:
            continue
        if hasattr(obj, "components"):
            print(f"== {type(obj).__name__} components ==")
            for name, comp in getattr(obj, "components", {}).items():
                if isinstance(comp, Module):
                    nbytes = _bytes_on_module_params(comp)
                    if not only_nonzero or nbytes > 0:
                        try:
                            dev = next(comp.parameters()).device
                        except StopIteration:
                            dev = "n/a"
                        print(f"  {name:<22} {type(comp).__name__:<34} cuda_bytes={_fmt_mib(nbytes)} dev={dev}")
        elif isinstance(obj, Module):
            nbytes = _bytes_on_module_params(obj)
            if not only_nonzero or nbytes > 0:
                print(f"== {type(obj).__name__} == cuda_bytes={_fmt_mib(nbytes)}")


# --------------------- CUDA cache nuking (runtime refs) ---------------------

def _maybe_nuke_attr(obj, name):
    try:
        val = getattr(obj, name, None)
    except Exception:
        return 0
    freed = 0
    if torch.is_tensor(val) and getattr(val, "is_cuda", False):
        try:
            setattr(obj, name, None)
            freed += val.nelement() * val.element_size()
        except Exception:
            pass
    return freed

def _unique_fname(dirpath: str, human_name: str, mod: Module, ext: str) -> str:
    safe = human_name.replace("/", "_").replace(" ", "_")
    uid  = f"{id(mod):x}"  # per-instance uniqueness
    return os.path.join(dirpath, f"{safe}-{uid}{ext}")

def _drop_runtime_cuda_refs(obj):
    """Best-effort: clear common 'previous_*' and cache attributes that pin VRAM."""
    patterns = (
        "previous_", "prev_", "cache", "kv_cache",
        "attn_cache", "hidden_states", "last_hidden_state",
        "embeddings", "k_cache", "v_cache",
    )
    freed = 0
    queue = [obj]
    visited = set()
    while queue:
        cur = queue.pop()
        oid = id(cur)
        if oid in visited:
            continue
        visited.add(oid)

        # modules: inspect __dict__ only (avoid recursion into parameters)
        d = getattr(cur, "__dict__", None)
        if isinstance(d, dict):
            for k in list(d.keys()):
                low = str(k).lower()
                if any(low.startswith(p) or low.endswith(p) for p in patterns):
                    freed += _maybe_nuke_attr(cur, k)

        # nested containers
        if isinstance(cur, dict):
            for v in list(cur.values()):
                if isinstance(v, (Module, dict, list, tuple)):
                    queue.append(v)
        elif isinstance(cur, (list, tuple)):
            for v in list(cur):
                if isinstance(v, (Module, dict, list, tuple)):
                    queue.append(v)
        else:
            # recurse into submodules/components, not into tensors
            if isinstance(cur, Module):
                for ch in cur.children():
                    queue.append(ch)
            comps = getattr(cur, "components", None)
            if isinstance(comps, dict):
                for v in comps.values():
                    if isinstance(v, (Module, dict, list, tuple)):
                        queue.append(v)
    if freed:
        torch.cuda.empty_cache()
    return freed


# --------------------- Tensorizer-backed Offload ---------------------

def _choose_parent_dir(disk_dir: str | None, prefer_tmpfs: bool = True) -> str | None:
    if disk_dir:
        return disk_dir
    if prefer_tmpfs:
        shm = Path("/dev/shm/astria-offload")
        try:
            shm.mkdir(parents=True, exist_ok=True)
            return str(shm)
        except Exception:
            pass
    if _ASTRIA_CACHE_DIR:
        try:
            Path(_ASTRIA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            return _ASTRIA_CACHE_DIR
        except Exception:
            pass
    return None


def _serialize_module_tensorizer(mod: Module, path: str):
    with open(path, "wb+") as f:
        s = TensorSerializer(f)
        s.write_module(mod, include_non_persistent_buffers=False)
        s.close()

def _deserialize_into_module_tensorizer(mod: Module, path: str, device: torch.device):
    want_cuda = (device.type == "cuda")
    with open(path, "rb") as f, TensorDeserializer(
        f, device=device, dtype=None, plaid_mode=want_cuda
    ) as td:
        td.load_into_module(mod)

def _serialize_module_safetensors(mod: Module, path: str):
    sd = {}
    for k, v in mod.state_dict().items():
        t = v.detach()
        if t.is_cuda:
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        sd[k] = t
    st_save_file(sd, path, metadata={"astria": "offload"})

def _deserialize_into_module_safetensors(mod: Module, path: str, device: torch.device):
    sd = st_load_file(path, device="cpu")
    mod.load_state_dict(sd, strict=True)
    mod.to(device, non_blocking=True)

def _serialize_module_torchsave(mod: Module, path: str):
    # Still use state_dict (pickled full objects are fast but brittle and unsafe).
    sd = mod.state_dict()
    torch.save(sd, path, _use_new_zipfile_serialization=True)

def _deserialize_into_module_torchsave(mod: Module, path: str, device: torch.device):
    sd = torch.load(path, map_location="cpu")
    mod.load_state_dict(sd, strict=True)
    mod.to(device, non_blocking=True)

def _get_serializer_impl(serializer: str):
    ser = (serializer or "").lower()
    if ser == "tensorizer" and _HAS_TENSORIZER:
        return _serialize_module_tensorizer, _deserialize_into_module_tensorizer, ".tensors"
    if ser == "safetensors" and _HAS_SAFETENSORS:
        return _serialize_module_safetensors, _deserialize_into_module_safetensors, ".safetensors"
    return _serialize_module_torchsave, _deserialize_into_module_torchsave, ".pt"


def _evict_module_to_meta(mod: Module):
    """Free VRAM immediately by moving CUDA params/buffers to meta (shape-preserving, zero VRAM)."""
    for p in mod.parameters(recurse=True):
        if getattr(p, "is_cuda", False):
            try:
                p.data = torch.empty_like(p.data, device="meta")
            except Exception:
                # fallback: tiny CPU tensor (keeps dtype/shape? no; last resort)
                p.data = torch.empty(0, dtype=p.dtype, device="cpu")
    for name, b in list(mod.named_buffers(recurse=True)):
        if getattr(b, "is_cuda", False):
            try:
                new_b = torch.empty_like(b, device="meta")
            except Exception:
                new_b = torch.empty(0, dtype=b.dtype, device="cpu")
            try:
                mod.register_buffer(name, new_b, persistent=True)
            except Exception:
                setattr(mod, name, new_b)


def _bytes_on_module(mod: Module) -> int:
    try:
        return sum(t.nelement() * t.element_size() for t in mod.state_dict().values())
    except Exception:
        return 0


@contextmanager
def offload_pipes_to_disk(
    *objs,
    disk_dir: str | None = None,
    device: str = "cuda",
    include: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = (),
    keep_on_cuda: tuple[str, ...] = ("text_encoder", "text_encoder_2", "clip_vision_model", "tokenizer", "tokenizer_2"),
    min_module_bytes: int = 1 << 20,  # 1 MiB
    parallel_io: int = 4,
    serializer: str = "tensorizer",   # "tensorizer" | "safetensors" | "torch"
    prefer_tmpfs: bool = True,
    cleanup: bool = True,
    debug: bool = True,
    debug_top_tensors: int = 12,
    record_memory_history: bool = False,
    snapshot_path: str | None = None,
):
    """
    Transiently offload big modules from Diffusers pipelines (or raw nn.Module objects)
    to disk. **Serializes from GPU**; **evicts to meta** immediately to free VRAM.
    Restores **straight to CUDA** (Tensorizer plaid mode) in ascending-size order.

    Also scrubs runtime CUDA caches (`previous_*`, `*_cache`, KV caches) so no
    stray refs pin VRAM (e.g., in `self.last_pipe`).
    """
    if debug:
        print("OFFLOADING PIPES TO DISK! (manual, no hooks)")

    onload_dev = torch.device(device) if not isinstance(device, torch.device) else device
    if onload_dev.type != "cuda":
        raise ValueError("offload_pipes_to_disk expects a CUDA device to restore to")

    start = time.time()
    parent = _choose_parent_dir(disk_dir, prefer_tmpfs=prefer_tmpfs)
    tmp = tempfile.mkdtemp(dir=parent, prefix="manual-offload-")
    created_tmp = True

    if record_memory_history:
        try:
            ok = _enable_cuda_history()
            if debug:
                print("CUDA memory history recording", "ENABLED" if ok else "UNAVAILABLE")
        except Exception as e:
            if debug:
                print(f"Could not enable CUDA memory history: {e}")

    def _want(name: str, mod: Module) -> bool:
        if not isinstance(mod, Module):
            return False
        if include is not None and not any(name.startswith(i) for i in include):
            return False
        if any(name.startswith(k) for k in keep_on_cuda):
            return False
        if any(name.startswith(e) for e in exclude):
            return False
        if _bytes_on_module(mod) < min_module_bytes:
            return False
        return True

    to_offload: list[tuple[str, Module, str, int]] = []  # (name, module, path, size_bytes)

    # Scan inputs
    for obj in objs:
        if obj is None:
            continue

        # Aggressively drop runtime caches that keep CUDA tensors alive
        _drop_runtime_cuda_refs(obj)

        # inside the DiffusionPipeline branch
        if isinstance(obj, DiffusionPipeline) and hasattr(obj, "components"):
            for cname, comp in obj.components.items():
                if isinstance(comp, Module) and _want(cname, comp):
                    size = _bytes_on_module(comp)
                    if size < min_module_bytes:
                        continue
                    ext  = ".tensors" if (serializer.lower() == "tensorizer" and _HAS_TENSORIZER) else (
                        ".safetensors" if (serializer.lower() == "safetensors" and _HAS_SAFETENSORS) else ".pt"
                    )
                    human = f"{type(obj).__name__}.{cname}"
                    path  = _unique_fname(tmp, human, comp, ext)
                    to_offload.append((human, comp, path, size))
            continue

        # plain Module
        if isinstance(obj, Module) and _want(type(obj).__name__, obj):
            size = _bytes_on_module(obj)
            if size >= min_module_bytes:
                ext  = ".tensors" if (serializer.lower() == "tensorizer" and _HAS_TENSORIZER) else (
                    ".safetensors" if (serializer.lower() == "safetensors" and _HAS_SAFETENSORS) else ".pt"
                )
                human = type(obj).__name__
                path  = _unique_fname(tmp, human, obj, ext)
                to_offload.append((human, obj, path, size))
            # no continue; still check nested attrs

        # nested attributes (also fix a stray ')' that was in the previous version)
        for attr in ("model", "transformer", "unet", "vae", "text_encoder", "text_encoder_2", "clip_vision_model"):
            sub = getattr(obj, attr, None)
            if isinstance(sub, Module) and _want(attr, sub):
                size = _bytes_on_module(sub)
                if size < min_module_bytes:
                    continue
                ext  = ".tensors" if (serializer.lower() == "tensorizer" and _HAS_TENSORIZER) else (
                    ".safetensors" if (serializer.lower() == "safetensors" and _HAS_SAFETENSORS) else ".pt"
                )
                human = f"{type(obj).__name__}.{attr}"
                path  = _unique_fname(tmp, human, sub, ext)
                to_offload.append((human, sub, path, size))

    # Serializer implementations
    _save_one, _load_one, _ext = _get_serializer_impl(serializer)

    # Sort saves by descending size (better disk throughput) and restores by ascending size (lower peak VRAM)
    to_offload.sort(key=lambda x: x[3], reverse=True)

    # 1) Write from GPU, then evict to meta (no bulk .to('cpu'))
    def _save_then_evict(entry):
        name, mod, path, _size = entry
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _save_one(mod, path)
        _evict_module_to_meta(mod)  # frees VRAM now
        setattr(mod, "__offloaded_path__", path)
        return name

    try:
        if debug:
            print(f"Planning to offload {len(to_offload)} module(s) to {tmp}")
        if to_offload:
            # First pass: drop any lingering runtime CUDA tensors again (esp. last_pipe cached tensors)
            for obj in objs:
                _drop_runtime_cuda_refs(obj)

            with ThreadPoolExecutor(max_workers=max(1, parallel_io)) as ex:
                futs = [ex.submit(_save_then_evict, x) for x in to_offload]
                for fu in as_completed(futs):
                    _ = fu.result()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if debug:
            print(f"DONE OFFLOADING in {time.time() - start:.3f}s")
            report_cuda_memory(onload_dev)
            summarize_objects_cuda(objs, only_nonzero=True)
            gc.collect()
            list_live_cuda_tensors(limit=debug_top_tensors)
            if snapshot_path:
                try:
                    torch.cuda.synchronize()
                    torch.cuda.memory._dump_snapshot(snapshot_path)
                    print(f"CUDA memory snapshot written to: {snapshot_path}")
                except Exception as e:
                    print(f"Could not dump CUDA memory snapshot: {e}")

        yield

    finally:
        # Restore order: smallest first to reduce peak memory pressure
        to_offload.sort(key=lambda x: x[3])

        # Extra belt-and-suspenders: drop caches before restore
        for obj in objs:
            _drop_runtime_cuda_refs(obj)
        torch.cuda.empty_cache()

        def _restore(entry):
            name, mod, path, _size = entry
            try:
                _load_one(mod, path, onload_dev)
                try:
                    mod.to(onload_dev, non_blocking=True)
                except Exception:
                    pass
                try:
                    delattr(mod, "__offloaded_path__")
                except Exception:
                    pass
                return name
            except Exception as e:
                print(f"[restore] FAILED {name} from {path}: {e}", file=sys.stderr)
                raise

        restored = 0
        if to_offload:
            with ThreadPoolExecutor(max_workers=max(1, parallel_io)) as ex:
                futs = [ex.submit(_restore, x) for x in to_offload]
                for fu in as_completed(futs):
                    _ = fu.result()
                    restored += 1

        # Remove any Accelerate hooks (harmless if none)
        for _, mod, _, _ in to_offload:
            try:
                remove_hook_from_submodules(mod)
            except Exception:
                pass

        # Cleanup temp folder
        if cleanup and created_tmp:
            shutil.rmtree(tmp, ignore_errors=True)

        if debug:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("AFTER RESTORE:")
            print(f"(cleanup) restored_modules={restored} tmp_dir={tmp}")
            report_cuda_memory(onload_dev)
