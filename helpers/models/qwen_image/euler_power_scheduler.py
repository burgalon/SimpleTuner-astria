# flow_match_euler_power_scheduler.py
import numpy as np
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler
)
from diffusers.configuration_utils import register_to_config

class FlowMatchEulerDiscreteSchedulerPower(FlowMatchEulerDiscreteScheduler):
    @register_to_config
    def __init__(self, *args, use_power_sigmas: bool=False, power_exponent: float=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if sum([
            self.config.use_beta_sigmas,
            self.config.use_exponential_sigmas,
            self.config.use_karras_sigmas,
            use_power_sigmas,
        ]) > 1:
            raise ValueError("Choose only one of {beta, exponential, karras, power} sigma options.")

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: torch.device | str | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ):
        """
        Keep the same signature so the pipeline can pass sigmas/timesteps.
        """
        # Let the base class compute the default schedule (incl. shifting/terminal/inversion)
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )

        # If power mode is off, we’re done.
        if not getattr(self.config, "use_power_sigmas", False):
            return

        if any([self.config.use_karras_sigmas, self.config.use_exponential_sigmas, self.config.use_beta_sigmas]):
            raise ValueError("`use_power_sigmas` cannot be combined with karras/exponential/beta.")

        # Current sigmas include the terminal element at the end. We reindex only the main path.
        base = self.sigmas[:-1]  # shape [N]
        N = base.shape[0]
        if N < 2:
            return  # nothing to warp

        p = float(getattr(self.config, "power_exponent", 1.0))
        r = np.linspace(0.0, 1.0, N, dtype=np.float64) ** p
        idx = np.round(r * (N - 1)).astype(np.int64)
        idx = np.clip(idx, 0, N - 1)

        new_sigmas = base[idx]

        # Rebuild scheduler tensors
        terminal = self.sigmas[-1:].clone()
        self.sigmas = torch.cat([new_sigmas, terminal]).to(device=self.sigmas.device, dtype=self.sigmas.dtype)
        # Recompute timesteps to stay consistent with FM Euler notion: t = sigma * T
        self.timesteps = (self.sigmas[:-1] * self.config.num_train_timesteps).to(
            device=self.timesteps.device, dtype=self.timesteps.dtype
        )
        self._step_index = None
        self._begin_index = None
