# tensor_bundles.py
# Drop next to your inference code and: `from tensor_bundles import load_or_tensorize_bundle`

import json, os, importlib
from pathlib import Path
from typing import Any, Dict, Optional

from contextlib import contextmanager

import torch.nn as nn
import torch.nn.init as init

from omegaconf import OmegaConf
from astria.seedvr2.common.config import load_config, create_object

import torch
from diffusers import DiffusionPipeline, ModelMixin, ConfigMixin
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig

from tensorizer import TensorSerializer, TensorDeserializer, utils
from filelock import FileLock, Timeout

from astria_utils import CACHE_DIR

BUNDLE_DIRNAME = "tensorizer_bundle"

# ---------- helpers ----------
def _class_path(obj_or_cls) -> str:
    c = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
    return f"{c.__module__}:{c.__name__}"

def _import_class(path: str):
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)

def _is_learned_module(x) -> bool:
    return isinstance(x, torch.nn.Module) and any(p.numel() for p in x.parameters())

def _component_dtype(module: torch.nn.Module) -> Optional[str]:
    try:
        for t in module.state_dict().values():
            return str(t.dtype).replace("torch.", "")
    except Exception:
        pass
    return None

def _bundle_dir(model_dir: str) -> Path:
    return Path(model_dir) / BUNDLE_DIRNAME

def _cfg_dir(bundle_dir: Path, name: str) -> Path:
    return bundle_dir / "configs" / name

def _tensor_path(bundle_dir: Path, name: str) -> Path:
    return bundle_dir / f"{name}.tensors"

def _manifest_path(bundle_dir: Path) -> Path:
    return bundle_dir / "manifest.json"

def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic on POSIX

@contextmanager
def suppress_parameter_initialization():
    """
    Temporarily disable expensive torch.nn.init calls and common reset_parameters
    to avoid Kaiming/Xavier/normal init during module construction.
    """
    init_names = [
        "kaiming_uniform_", "kaiming_normal_",
        "xavier_uniform_", "xavier_normal_",
        "normal_", "uniform_", "constant_",
        "zeros_", "ones_",
        "trunc_normal_", "truncated_normal_",
    ]
    saved_inits = {n: getattr(init, n, None) for n in init_names if hasattr(init, n)}
    def _noop(t, *args, **kwargs):  # return tensor unchanged
        return t
    for n in saved_inits:
        setattr(init, n, _noop)

    # Common modules that call reset_parameters() in __init__
    classes = [
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.Embedding,
    ]
    saved_resets = {}
    for cls in classes:
        if hasattr(cls, "reset_parameters"):
            saved_resets[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self: None  # type: ignore

    try:
        yield
    finally:
        for n, fn in saved_inits.items():
            setattr(init, n, fn)
        for cls, fn in saved_resets.items():
            cls.reset_parameters = fn  # type: ignore



# ---------- serialize (one-time) ----------
def tensorize_pipeline(model_dir: str, force: bool = False, torch_dtype: Optional[torch.dtype] = None) -> None:
    """Create a tensorizer bundle for any Diffusers pipeline at model_dir."""
    model_dir = str(model_dir)
    bundle_dir = _bundle_dir(model_dir)
    man_path = _manifest_path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "configs").mkdir(parents=True, exist_ok=True)

    if man_path.exists() and not force:
        return
    print('Building tensorizer bundle...')

    pipe = DiffusionPipeline.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch_dtype)

    # Prefer the canonical component map when available
    components: Dict[str, Any] = getattr(pipe, "components", None) or {
        k: v for k, v in vars(pipe).items() if not k.startswith("_")
    }

    role2meta: Dict[str, Dict[str, Optional[str]]] = {}
    for name, comp in components.items():
        if comp is None:
            continue

        # Learned modules -> write config + tensors
        if _is_learned_module(comp):
            cfgdir = _cfg_dir(bundle_dir, name)
            cfgdir.mkdir(parents=True, exist_ok=True)

            # Save config for both diffusers (ConfigMixin) and transformers (PretrainedConfig)
            cfg = getattr(comp, "config", None)
            if isinstance(cfg, (PretrainedConfig,)):
                cfg.save_pretrained(str(cfgdir))
            elif isinstance(comp, (ConfigMixin, ModelMixin)) and hasattr(comp, "save_config"):
                comp.save_config(str(cfgdir))
            else:
                # very rare: dump raw dict
                with open(cfgdir / "config.json", "w") as f:
                    json.dump(getattr(cfg, "to_dict", lambda: cfg)(), f)

            # Serialize tensors
            tpath = _tensor_path(bundle_dir, name)
            with open(tpath, "wb+") as f:
                s = TensorSerializer(f)
                s.write_module(comp, include_non_persistent_buffers=False)
                s.close()

            role2meta[name] = {"type": _class_path(comp), "kind": "module", "dtype": _component_dtype(comp)}
            continue

        # Everything config-only with save_pretrained (tokenizers, schedulers, processors, feature extractors)
        if hasattr(comp, "save_pretrained"):
            comp.save_pretrained(str(_cfg_dir(bundle_dir, name)))
            role2meta[name] = {"type": _class_path(comp), "kind": "pretrained", "dtype": None}

    manifest = {
        "pipeline_class": _class_path(pipe),
        "components": role2meta,
        "lib_versions": {
            "diffusers": __import__("diffusers").__version__,
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
        },
    }
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # free RAM
    del pipe
    torch.cuda.empty_cache()


# ---------- load (fast path) ----------
def _load_module_from_bundle(bundle_dir: Path, name: str, device: torch.device, dtype: Optional[torch.dtype]):
    # Decide diffusers vs transformers config loader
    # We don't know the exact class until manifest is read.
    # We'll reconstruct using the saved class path:
    manifest = json.loads(_manifest_path(bundle_dir).read_text())
    cls = _import_class(manifest["components"][name]["type"])
    cfgdir = _cfg_dir(bundle_dir, name)

    # Build the (empty) module
    cfg = None
    if hasattr(cls, "load_config"):            # diffusers ModelMixin
        cfg = cls.load_config(str(cfgdir))
        with utils.no_init_or_tensor():
            module = getattr(cls, "from_config", cls)(cfg)
    else:                                       # transformers PreTrainedModel
        cfg = AutoConfig.from_pretrained(str(cfgdir))
        with utils.no_init_or_tensor():
            module = cls(cfg)                   # PreTrainedModel accepts config in ctor

    # Stream tensors straight onto device
    tpath = _tensor_path(bundle_dir, name)
    with open(tpath, "rb") as f, TensorDeserializer(f, device=device, dtype=dtype) as td:
        td.load_into_module(module)
    return module


def load_bundle(model_dir: str, device: torch.device = utils.get_device(), dtype_overrides: Optional[Dict[str, torch.dtype]] = None):
    """Rebuild the pipeline from a tensorizer bundle."""
    bundle_dir = _bundle_dir(model_dir)
    manifest = json.loads(_manifest_path(bundle_dir).read_text())
    pipe_cls = _import_class(manifest["pipeline_class"])
    comps: Dict[str, Any] = {}

    for name, meta in manifest["components"].items():
        if meta["kind"] == "module":
            target_dtype = (dtype_overrides or {}).get(name, None)
            comps[name] = _load_module_from_bundle(bundle_dir, name, device, target_dtype)
        else:
            # config-only bits: tokenizers, schedulers, processors, etc.
            cls = _import_class(meta["type"])
            comps[name] = cls.from_pretrained(str(_cfg_dir(bundle_dir, name)), local_files_only=True)

    pipe = pipe_cls(**comps).to(device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


# ---------- convenience ----------
def bundle_exists(model_dir: str) -> bool:
    man = _manifest_path(_bundle_dir(model_dir))
    if not man.exists():
        return False
    try:
        m = json.loads(man.read_text())
        return bool(m.get("components"))
    except Exception:
        return False


def load_or_tensorize_bundle(model_dir: str, device: torch.device, default_dtype: Optional[torch.dtype] = None,
                             dtype_overrides: Optional[Dict[str, torch.dtype]] = None, force_rebuild: bool = False):
    """Try fast path; otherwise build bundle once and load."""
    lock = FileLock(str(_bundle_dir(model_dir) / ".bundle.lock"), timeout=60)
    with lock:
        if not bundle_exists(model_dir) or force_rebuild:
            tensorize_pipeline(model_dir, force=force_rebuild, torch_dtype=default_dtype)
    return load_bundle(model_dir, device=device, dtype_overrides=dtype_overrides or {})


# --- SeedVR2-specific paths ---
def _seedvr2_bundle_dir(ckpt_path: str) -> Path:
    ckpt = Path(ckpt_path).resolve()
    return ckpt.parent / f"{ckpt.stem}.tz"             # e.g., /cache/seedvr2_ema_7b.tz

def _seedvr2_manifest_path(bundle_dir: Path) -> Path:
    return bundle_dir / "manifest.seedvr2.json"

def _yaml_dump(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(content)))

def _yaml_load(path: Path):
    return OmegaConf.load(str(path))

# ---------- one-time tensorization ----------
def seedvr2_tensorize(
    config_yaml: str,
    dit_ckpt: str,
    vae_ckpt: str,
    *,
    force: bool = False,
) -> Path:
    bundle_dir = _seedvr2_bundle_dir(dit_ckpt)
    man_path   = _seedvr2_manifest_path(bundle_dir)
    cfg_dir    = bundle_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    if man_path.exists() and not force:
        return bundle_dir

    from omegaconf import OmegaConf
    from astria.seedvr2.common.config import load_config, create_object

    cfg = load_config(str(config_yaml))

    # Build skeletons without expensive init
    from contextlib import ExitStack
    with ExitStack() as stack:
        try:
            stack.enter_context(suppress_parameter_initialization())
        except Exception:
            pass

        with torch.device("cpu"):
            dit = create_object(cfg.dit.model)
            vae = create_object(cfg.vae.model)

    # Load checkpoints on CPU
    dit_sd = torch.load(dit_ckpt, map_location="cpu", mmap=True)
    vae_sd = torch.load(vae_ckpt, map_location="cpu", mmap=True)
    dit.load_state_dict(dit_sd, strict=True)
    vae.load_state_dict(vae_sd, strict=True)

    # ---- cast to final dtypes BEFORE serializing (smaller, faster) ----
    dit_dtype = torch.bfloat16
    vae_dtype = getattr(torch, str(cfg.vae.dtype))
    dit.to(dtype=dit_dtype)
    vae.to(dtype=vae_dtype)

    # Save minimal configs for reconstruction
    _yaml_dump(cfg_dir / "dit_model.yaml", OmegaConf.to_container(cfg.dit.model, resolve=True))
    _yaml_dump(cfg_dir / "vae_model.yaml", OmegaConf.to_container(cfg.vae.model, resolve=True))
    (cfg_dir / "vae_dtype.txt").write_text(str(cfg.vae.dtype))

    # Tensors (must be wb+)
    with open(bundle_dir / "dit.tensors", "wb+") as f:
        s = TensorSerializer(f)
        s.write_module(dit, include_non_persistent_buffers=False)
        s.close()
    with open(bundle_dir / "vae.tensors", "wb+") as f:
        s = TensorSerializer(f)
        s.write_module(vae, include_non_persistent_buffers=False)
        s.close()

    manifest = {
        "kind": "seedvr2",
        "components": {
            "dit": {"type": _class_path(dit), "dtype": "bfloat16"},
            "vae": {"type": _class_path(vae), "dtype": str(cfg.vae.dtype)},
        },
        "lib_versions": {"torch": torch.__version__},
    }
    _atomic_write_json(man_path, manifest)

    del dit, vae, dit_sd, vae_sd
    torch.cuda.empty_cache()
    return bundle_dir

# tensor_bundles.py (SeedVR2 path)

def load_seedvr2_bundle(dit_ckpt: str, device: torch.device = utils.get_device()):
    device = torch.device(device)  # accept str or torch.device
    want_cuda = device.type == "cuda"
    bundle_dir = _seedvr2_bundle_dir(dit_ckpt)
    _ = json.loads(_seedvr2_manifest_path(bundle_dir).read_text())

    # --- DiT ---
    dit_cfg = _yaml_load(bundle_dir / "configs" / "dit_model.yaml")
    with suppress_parameter_initialization(), utils.no_init_or_tensor():
        dit = create_object(dit_cfg)   # build skeleton (no heavy init)
    with open(bundle_dir / "dit.tensors", "rb") as f, TensorDeserializer(
        f, device=device, dtype=None, plaid_mode=want_cuda
    ) as td:
        td.load_into_module(dit)
    dit.to(device, non_blocking=True)  # move any CPU leftovers

    # --- VAE ---
    vae_cfg = _yaml_load(bundle_dir / "configs" / "vae_model.yaml")
    with suppress_parameter_initialization(), utils.no_init_or_tensor():
        vae = create_object(vae_cfg)
    with open(bundle_dir / "vae.tensors", "rb") as f, TensorDeserializer(
        f, device=device, dtype=None, plaid_mode=want_cuda
    ) as td:
        td.load_into_module(vae)
    vae.to(device, non_blocking=True)

    # Optional: sanity check – ensure no meta tensors remain anywhere
    for t in list(dit.state_dict().values()) + list(vae.state_dict().values()):
        if getattr(t, "is_meta", False):
            raise RuntimeError("meta tensor left after deserialization")

    return dit, vae


def seedvr2_load_or_tensorize(
    config_yaml: str,
    dit_ckpt: str,
    vae_ckpt: str,
    *,
    device: torch.device = utils.get_device(),
    force_rebuild: bool = False,
):
    bundle_dir = _seedvr2_bundle_dir(dit_ckpt)
    man_path = _seedvr2_manifest_path(bundle_dir)
    if (not man_path.exists()) or force_rebuild:
        seedvr2_tensorize(
            config_yaml, dit_ckpt, vae_ckpt,
            force=force_rebuild,
        )

    return load_seedvr2_bundle(dit_ckpt, device=device)
