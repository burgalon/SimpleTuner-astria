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

"""
Utility functions for creating schedules and samplers from config.
"""

from typing import TYPE_CHECKING
import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from astria.seedvr2.common.diffusion.samplers.base import Sampler
    from astria.seedvr2.common.diffusion.schedules.base import Schedule
    from astria.seedvr2.common.diffusion.schedules.lerp import LinearInterpolationSchedule
    from astria.seedvr2.common.diffusion.timesteps.base import SamplingTimesteps


def create_schedule_from_config(
    config: DictConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> 'Schedule':
    """
    Create a schedule from configuration.
    """
    from astria.seedvr2.common.diffusion.schedules.lerp import LinearInterpolationSchedule
    if config.type == "lerp":
        return LinearInterpolationSchedule(T=config.get("T", 1.0))

    raise NotImplementedError


def create_sampler_from_config(
    config: DictConfig,
    schedule: 'Schedule',
    timesteps: 'SamplingTimesteps',
) -> 'Sampler':
    """
    Create a sampler from configuration.
    """
    from astria.seedvr2.common.diffusion.samplers.euler import EulerSampler
    if config.type == "euler":
        return EulerSampler(
            schedule=schedule,
            timesteps=timesteps,
            prediction_type=config.prediction_type,
        )
    raise NotImplementedError


def create_sampling_timesteps_from_config(
    config: DictConfig,
    schedule: 'Schedule',
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> 'SamplingTimesteps':
    from astria.seedvr2.common.diffusion.timesteps.sampling.trailing import UniformTrailingSamplingTimesteps
    if config.type == "uniform_trailing":
        return UniformTrailingSamplingTimesteps(
            T=schedule.T,
            steps=config.steps,
            shift=config.get("shift", 1.0),
            device=device,
        )
    raise NotImplementedError
