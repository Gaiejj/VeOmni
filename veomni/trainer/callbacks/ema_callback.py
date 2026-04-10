# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EMA (Exponential Moving Average) callback for training.

Maintains shadow copies of model parameters as exponential moving averages.
Compatible with FSDP2 sharded parameters - EMA operates on local shards only,
requiring no additional communication.
"""

from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn as nn

from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer

logger = helper.create_logger(__name__)


class EMACallback(Callback):
    """Exponential Moving Average callback.

    Maintains EMA shadow weights that track the model parameters with
    exponential decay. Works with FSDP2 by operating on local shards.

    Args:
        trainer: The trainer instance.
        decay: EMA decay factor (default 0.9999).
        warmup_steps: Number of steps to linearly warmup decay from 0 to target (default 0).
        update_after_step: Start EMA updates after this many steps (default 0).
        update_every: Update EMA every N steps (default 1).
    """

    def __init__(
        self,
        trainer: "BaseTrainer",
        decay: float = 0.9999,
        warmup_steps: int = 0,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        super().__init__(trainer)
        self.target_decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        self.update_every = update_every

        # Initialize EMA shadow parameters as detached clones of model params.
        # With FSDP2, parameters are sharded DTensors - we clone the local shard.
        self.ema_params: list[torch.Tensor] = []
        self.model_params: list[nn.Parameter] = []

        model = self.trainer.model
        for param in model.parameters():
            if param.requires_grad:
                # .data gives us the local shard in FSDP2
                self.ema_params.append(param.data.clone().detach())
                self.model_params.append(param)

        logger.info_rank0(
            f"EMA initialized with {len(self.ema_params)} parameters, "
            f"decay={decay}, warmup_steps={warmup_steps}, "
            f"update_after_step={update_after_step}, update_every={update_every}"
        )

    def _get_decay(self, step: int) -> float:
        """Get current decay value, with optional warmup."""
        if step < self.update_after_step:
            return 0.0
        if self.warmup_steps > 0:
            effective_step = step - self.update_after_step
            if effective_step < self.warmup_steps:
                return self.target_decay * (effective_step / self.warmup_steps)
        return self.target_decay

    @torch.no_grad()
    def _update(self, step: int):
        """Update EMA parameters."""
        decay = self._get_decay(step)
        if decay == 0.0:
            return
        for ema_p, model_p in zip(self.ema_params, self.model_params):
            # lerp_: ema_p = ema_p + (1 - decay) * (model_p.data - ema_p)
            #       = decay * ema_p + (1 - decay) * model_p.data
            ema_p.lerp_(model_p.data, 1.0 - decay)

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        """Update EMA after each optimizer step."""
        if state.global_step % self.update_every == 0:
            self._update(state.global_step)

    @torch.no_grad()
    def swap_to_ema(self):
        """Swap model weights with EMA weights (for evaluation)."""
        for ema_p, model_p in zip(self.ema_params, self.model_params):
            tmp = model_p.data.clone()
            model_p.data.copy_(ema_p)
            ema_p.copy_(tmp)

    @torch.no_grad()
    def swap_from_ema(self):
        """Swap back from EMA weights to training weights.

        This is the same operation as swap_to_ema since the swap is symmetric.
        """
        self.swap_to_ema()

    def state_dict(self) -> Dict[str, Any]:
        """Return EMA state for checkpointing."""
        return {
            "ema_params": [p.cpu() for p in self.ema_params],
            "target_decay": self.target_decay,
            "warmup_steps": self.warmup_steps,
            "update_after_step": self.update_after_step,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Load EMA state from checkpoint."""
        saved_params = state["ema_params"]
        if len(saved_params) != len(self.ema_params):
            logger.warning_rank0(
                f"EMA param count mismatch: saved {len(saved_params)} vs current {len(self.ema_params)}"
            )
            return
        for ema_p, saved_p in zip(self.ema_params, saved_params):
            ema_p.copy_(saved_p.to(ema_p.device))
        logger.info_rank0("EMA state loaded successfully")
