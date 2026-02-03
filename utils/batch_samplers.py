import os
import random
from collections import defaultdict
from typing import Iterator, List, Sequence

import torch


def _get_dist_env():
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
    except Exception:
        world_size, rank = 1, 0
    return world_size, rank


class UniqueConceptBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Batch sampler that enforces *unique concept ids per batch*.

    Why: For concept-level targets (ds003825), many trials share the same target vector.
    Standard InfoNCE assumes a 1-1 mapping within batch (diagonal positives). If a batch
    contains duplicated targets, the supervision becomes ambiguous and training can stall
    near the random baseline (~log(batch_size)).

    This sampler groups dataset indices by concept id and samples at most 1 item per concept
    in each batch. It is rank-aware under torchrun (WORLD_SIZE/RANK) to avoid identical
    concept ordering across ranks.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        # Expect ds003825 backend fields.
        if not hasattr(dataset, "backend") or getattr(dataset, "backend") != "ds003825":
            raise ValueError("UniqueConceptBatchSampler requires TripletDataset backend == 'ds003825'")
        if not hasattr(dataset, "ds_concept_ids") or not hasattr(dataset, "indices"):
            raise ValueError("Dataset missing ds003825 fields ds_concept_ids/indices")

        # Build mapping: concept_id -> list of dataset-local indices (0..len-1)
        concept_to_items = defaultdict(list)
        for local_i, trial_index in enumerate(dataset.indices):
            concept = int(dataset.ds_concept_ids[int(trial_index)])
            concept_to_items[concept].append(local_i)

        self._concept_to_items = dict(concept_to_items)
        self._concepts: List[int] = sorted(self._concept_to_items.keys())

    def __iter__(self) -> Iterator[List[int]]:
        world_size, rank = _get_dist_env()

        # Rank-aware shuffle of concept list.
        concepts = list(self._concepts)
        if self.shuffle:
            rng = random.Random(self.seed + rank)
            rng.shuffle(concepts)

        # Partition concepts across ranks deterministically.
        if world_size > 1:
            concepts = concepts[rank::world_size]

        # Sample one trial per concept to build batches.
        rng_pick = random.Random(self.seed + 10_000 + rank)
        batch: List[int] = []
        for concept in concepts:
            items = self._concept_to_items[concept]
            batch.append(rng_pick.choice(items) if self.shuffle else items[0])
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        world_size, rank = _get_dist_env()
        n = len(self._concepts)
        if world_size > 1:
            # Concepts assigned to this rank
            n = (n + world_size - 1 - rank) // world_size
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

