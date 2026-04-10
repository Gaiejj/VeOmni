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

"""Resolution bucket sampler for multi-resolution DiT training.

Groups training samples by resolution so that all samples in a batch
share the same spatial dimensions, enabling efficient batched training
across multiple resolutions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Sampler

from ...utils import logging


logger = logging.get_logger(__name__)


# Pre-defined resolution buckets for Wan2.1 training
# Format: (height, width) - frame count is handled separately
RESOLUTION_BUCKETS_256 = [
    (256, 256),
]

RESOLUTION_BUCKETS_480 = [
    (480, 832),  # 16:9
    (480, 640),  # 4:3
    (640, 480),  # 3:4
    (832, 480),  # 9:16
    (480, 480),  # 1:1
]

RESOLUTION_BUCKETS_720 = [
    (720, 1280),  # 16:9
    (720, 960),  # 4:3
    (960, 720),  # 3:4
    (1280, 720),  # 9:16
    (720, 720),  # 1:1
]

ALL_RESOLUTION_BUCKETS = RESOLUTION_BUCKETS_256 + RESOLUTION_BUCKETS_480 + RESOLUTION_BUCKETS_720


def find_nearest_bucket(height: int, width: int, buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Find the nearest resolution bucket for given dimensions.

    Uses a combined metric of aspect ratio difference and relative pixel count
    difference to select the best matching bucket.

    Args:
        height: Source image/video height.
        width: Source image/video width.
        buckets: List of (height, width) tuples defining available buckets.

    Returns:
        The (height, width) tuple of the closest matching bucket.
    """
    aspect_ratio = width / height
    best_bucket = buckets[0]
    best_diff = float("inf")
    for bh, bw in buckets:
        bucket_ar = bw / bh
        ar_diff = abs(aspect_ratio - bucket_ar) / max(aspect_ratio, bucket_ar)  # relative AR difference
        pixel_diff = abs(height * width - bh * bw) / max(height * width, bh * bw)  # relative pixel difference
        diff = ar_diff + 0.5 * pixel_diff  # AR is more important than pixel count
        if diff < best_diff:
            best_diff = diff
            best_bucket = (bh, bw)
    return best_bucket


class ResolutionBucketBatchSampler(Sampler):
    """Batch sampler that groups samples by resolution bucket.

    Each batch contains only samples from the same resolution bucket,
    ensuring all tensors in a batch have the same spatial dimensions.

    Args:
        dataset_size: Total number of samples in the dataset.
        resolution_map: Mapping from sample index to (height, width).
        batch_size: Number of samples per batch.
        buckets: List of (height, width) tuples defining resolution buckets.
            Defaults to ``ALL_RESOLUTION_BUCKETS``.
        drop_last: Whether to drop the last incomplete batch per bucket.
        shuffle: Whether to shuffle within buckets and across batch order.
        seed: Random seed for shuffling.
        rank: Process rank for distributed sampling.
        world_size: Total number of processes for distributed sampling.
    """

    def __init__(
        self,
        dataset_size: int,
        resolution_map: Dict[int, Tuple[int, int]],
        batch_size: int,
        buckets: Optional[List[Tuple[int, int]]] = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.buckets = buckets or ALL_RESOLUTION_BUCKETS
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        # Group indices by bucket
        self.bucket_indices: Dict[Tuple[int, int], List[int]] = {b: [] for b in self.buckets}
        for idx, (h, w) in resolution_map.items():
            bucket = find_nearest_bucket(h, w, self.buckets)
            self.bucket_indices[bucket].append(idx)

        # Remove empty buckets
        self.bucket_indices = {k: v for k, v in self.bucket_indices.items() if len(v) > 0}

        # Log bucket distribution
        for bucket, indices in self.bucket_indices.items():
            logger.info(f"Bucket {bucket}: {len(indices)} samples")

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across epochs."""
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []
        for _bucket, indices in self.bucket_indices.items():
            indices = indices.copy()
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            # Distribute across ranks
            # Ensure all ranks get equal sample counts to prevent deadlock
            n_per_rank = len(indices) // self.world_size
            rank_indices = indices[self.rank :: self.world_size][:n_per_rank]

            # Create batches
            for i in range(0, len(rank_indices), self.batch_size):
                batch = rank_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:  # always drop incomplete batches for distributed safety
                    all_batches.append(batch)

        # Shuffle batch order across buckets so training sees varied resolutions
        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]

        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self.bucket_indices.values():
            n = len(indices) // self.world_size
            if self.drop_last or self.world_size > 1:
                # Always drop incomplete batches in distributed mode to prevent deadlock
                total += n // self.batch_size
            else:
                total += math.ceil(n / self.batch_size)
        return total
