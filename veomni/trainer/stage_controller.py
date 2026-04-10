"""Multi-stage progressive training controller.

Manages training stage transitions for curriculum-style training,
where resolution, frame count, data, and learning rate change across stages.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils import logging


logger = logging.get_logger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single training stage."""

    name: str = "default"
    max_steps: int = 100000
    resolution: List[int] = field(default_factory=lambda: [256, 256])  # [height, width]
    max_frames: int = 1  # 1 for image-only, >1 for video
    lr: float = 1e-4
    lr_warmup_steps: int = 1000
    global_batch_size: int = -1  # -1 means use default from training args
    data_path: str = ""  # empty means use default from data args


class StageController:
    """Controls multi-stage progressive training.

    Tracks the current stage and manages transitions. When a stage completes
    (max_steps reached), signals the trainer to save a checkpoint, rebuild
    the dataloader with new resolution/data, and adjust the learning rate.

    Usage:
        controller = StageController(stages_config)
        while not controller.is_finished:
            # ... train one step ...
            controller.step()
            if controller.should_advance():
                controller.advance()
                # Rebuild dataloader, lr_scheduler, etc.
    """

    def __init__(self, stages: List[StageConfig]):
        if not stages:
            raise ValueError("StageController requires at least one stage.")
        self.stages = stages
        self.current_stage_idx = 0
        self.steps_in_current_stage = 0
        self._cumulative_steps: List[int] = []

        cumsum = 0
        for stage in stages:
            cumsum += stage.max_steps
            self._cumulative_steps.append(cumsum)

        self.total_steps = cumsum
        logger.info_rank0(f"StageController initialized with {len(stages)} stages, total {self.total_steps} steps")
        for i, stage in enumerate(stages):
            logger.info_rank0(
                f"  Stage {i}: {stage.name} — {stage.max_steps} steps, "
                f"resolution={stage.resolution}, frames={stage.max_frames}, lr={stage.lr}"
            )

    @property
    def current_stage(self) -> StageConfig:
        return self.stages[self.current_stage_idx]

    @property
    def is_finished(self) -> bool:
        return self.current_stage_idx >= len(self.stages)

    def global_step_to_stage(self, global_step: int) -> int:
        """Map global step to stage index."""
        for i, cum_steps in enumerate(self._cumulative_steps):
            if global_step < cum_steps:
                return i
        return len(self.stages)  # Past all stages

    def should_advance(self) -> bool:
        """Check if current stage is complete."""
        if self.is_finished:
            return False
        return self.steps_in_current_stage >= self.current_stage.max_steps

    def advance(self) -> Optional[StageConfig]:
        """Advance to next stage. Returns the new stage config or None if finished."""
        self.current_stage_idx += 1
        self.steps_in_current_stage = 0
        if self.is_finished:
            logger.info_rank0("All training stages completed.")
            return None
        logger.info_rank0(f"Advanced to stage {self.current_stage_idx}: {self.current_stage.name}")
        return self.current_stage

    def step(self):
        """Record one training step."""
        self.steps_in_current_stage += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_stage_idx": self.current_stage_idx,
            "steps_in_current_stage": self.steps_in_current_stage,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.current_stage_idx = state["current_stage_idx"]
        self.steps_in_current_stage = state["steps_in_current_stage"]
        if not self.is_finished:
            logger.info_rank0(
                f"Resumed at stage {self.current_stage_idx}: {self.current_stage.name}, "
                f"step {self.steps_in_current_stage}/{self.current_stage.max_steps}"
            )
