import inspect

from types import MethodType
from typing import Any, Dict, List

import torch

from diffusers.callbacks import PipelineCallback
from diffusers.configuration_utils import register_to_config

def enable_dynamic_fake_cfg(pipe):
    """
    Patches pipe.transformer.forward so the 'guidance' arg is rebuilt every call
    from pipe._guidance_scale, enabling callbacks to steer "fake" CFG mid-run.
    """
    tr = pipe.transformer

    # Safety: only patch models that actually use embedded guidance
    if not getattr(tr.config, "guidance_embeds", False):
        raise ValueError("This Flux transformer does not use embedded guidance (guidance_embeds=False).")

    # Idempotent
    if hasattr(tr, "_orig_forward"):
        return pipe

    orig_forward = tr.forward
    sig = inspect.signature(orig_forward)
    expects_guidance = "guidance" in sig.parameters

    def forward_with_dynamic_guidance(self, *args, **kwargs):
        # If the model doesn't accept 'guidance' or it's None, just run.
        if not expects_guidance or kwargs.get("guidance", None) is None:
            return orig_forward(*args, **kwargs)

        g = kwargs["guidance"]
        batch = g.shape[0]
        device = g.device
        dtype = g.dtype

        # Lazily allocate a 1-elem buffer we can fill() every call
        buf = getattr(self, "_guidance_buf", None)
        if (buf is None) or (buf.device != device) or (buf.dtype != dtype):
            buf = torch.empty(1, device=device, dtype=dtype)
            self._guidance_buf = buf

        # Read the *current* pipeline._guidance_scale (set by your callback)
        buf.fill_(float(getattr(pipe, "_guidance_scale", 1.0)))
        kwargs["guidance"] = buf.expand(batch)

        return orig_forward(*args, **kwargs)

    tr._orig_forward = orig_forward
    tr.forward = MethodType(forward_with_dynamic_guidance, tr)
    return pipe


def disable_dynamic_fake_cfg(pipe):
    tr = pipe.transformer
    if hasattr(tr, "_orig_forward"):
        tr.forward = tr._orig_forward
        delattr(tr, "_orig_forward")
        if hasattr(tr, "_guidance_buf"):
            delattr(tr, "_guidance_buf")


class FluxCFGCutoffCallback(PipelineCallback):
    """
    Adjust embedded (“fake”) CFG during Flux inference using cfg_mapping:
        [{"start": 0.0, "cfg": 3.5}, {"start": 0.35, "cfg": 3.4}, ...]
    'start' is a fraction of total steps; the new value applies after that index.
    """

    tensor_inputs: List[str] = []

    @register_to_config
    def __init__(self, cfg_mapping: List[Dict[str, float]]):
        # Satisfy base-ctor requirement (we don't actually use these)
        super().__init__(cutoff_step_ratio=1.0, cutoff_step_index=None)

        cfg_mapping = self._normalize_cfg_mapping(cfg_mapping)
        if not isinstance(cfg_mapping, list) or len(cfg_mapping) == 0:
            raise ValueError("`cfg_mapping` must be a non-empty list of {'start': float, 'cfg': float} dicts.")

        cleaned: List[Dict[str, float]] = []
        for i, it in enumerate(cfg_mapping):
            if not isinstance(it, dict) or "start" not in it or "cfg" not in it:
                raise ValueError(f"cfg_mapping[{i}] must contain 'start' and 'cfg'.")
            s = float(it["start"])
            if not (0.0 <= s <= 1.0):
                raise ValueError(f"cfg_mapping[{i}]['start'] must be in [0,1], got {s}.")
            cleaned.append({"start": s, "cfg": float(it["cfg"])})
        cleaned.sort(key=lambda d: d["start"])
        self._cfg = cleaned

        # Runtime state
        self._built: List[tuple] = []
        self._num_steps: int = -1
        self._ptr: int = 0

    # --- internals ---

    def _normalize_cfg_mapping(self, cfg_mapping):
        # tuple-of-list -> list
        if isinstance(cfg_mapping, tuple) and len(cfg_mapping) == 1 and isinstance(cfg_mapping[0], list):
            cfg_mapping = cfg_mapping[0]
        # list-of-list (single) -> list
        if isinstance(cfg_mapping, list) and len(cfg_mapping) == 1 and isinstance(cfg_mapping[0], list):
            cfg_mapping = cfg_mapping[0]
        return cfg_mapping

    def _resolve_total_steps(self, pipeline) -> int | None:
        """
        Determine #steps without touching pipeline.num_timesteps (some variants
        don't expose it). Prefer scheduler.timesteps length.
        """
        # 1) Private cache some pipelines keep
        v = getattr(pipeline, "_num_timesteps", None)
        if isinstance(v, int) and v > 0:
            return v

        # 2) Scheduler's current time grid (most reliable)
        sched = getattr(pipeline, "scheduler", None)
        if sched is not None:
            ts = getattr(sched, "timesteps", None)
            if ts is not None:
                try:
                    return len(ts)
                except Exception:
                    pass
            # occasional API shapes
            gt = getattr(sched, "get_timesteps", None)
            if callable(gt):
                try:
                    return len(gt())
                except Exception:
                    pass

        # 3) As a last resort, try not to hard-fail — return None so we no-op.
        return None

    def _build_schedule(self, num_steps: int) -> None:
        self._built = [(int(num_steps * e["start"]), e["cfg"]) for e in self._cfg]
        self._num_steps = num_steps
        self._ptr = 0

    def callback_fn(self, pipeline, step_index: int, timestep, callback_kwargs) -> Dict[str, Any]:
        # If this Flux build doesn’t use embedded guidance, nothing to do.
        if not getattr(pipeline.transformer.config, "guidance_embeds", False):
            return callback_kwargs

        total = self._resolve_total_steps(pipeline)
        if not total:
            # Can't determine schedule; gracefully skip this call.
            return callback_kwargs

        if self._num_steps != total:
            self._build_schedule(total)

        # We run at end of step k, so changes apply starting k+1.
        while self._ptr < len(self._built):
            start_idx, cfg_val = self._built[self._ptr]
            if (step_index + 1) >= start_idx:
                pipeline._guidance_scale = float(cfg_val)
                self._ptr += 1
            else:
                break

        return callback_kwargs