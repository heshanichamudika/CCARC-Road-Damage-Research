"""Country-Balanced Sampling (CBS) for CCARC Cross-Country Training.

Implements deterministic, reproducible, country-balanced batch and index samplers
compatible with PyTorch and Ultralytics data loaders. Supports all six RDD2022
sovereign countries, unifies China subsets (China_Drone and China_MotorBike),
strictly isolates held-out target country images, and handles unequal country
distributions and non-divisible batch sizes.
"""

from __future__ import annotations

from collections import defaultdict
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

try:
    from ultralytics.data.build import InfiniteDataLoader, seed_worker, PIN_MEMORY
except ImportError:
    from torch.utils.data import DataLoader as InfiniteDataLoader  # type: ignore[assignment,misc]
    seed_worker = None
    PIN_MEMORY = True

from src.data.loco_splitter import CHINA_SUBSETS, LOCO_COUNTRIES

# Canonical ordered prefixes for deterministic string matching (longest prefix first)
COUNTRY_PREFIX_MAP: Tuple[Tuple[str, str], ...] = (
    ("China_Drone_", "China"),
    ("China_MotorBike_", "China"),
    ("China_", "China"),
    ("United_States_", "United_States"),
    ("Czech_", "Czech"),
    ("India_", "India"),
    ("Japan_", "Japan"),
    ("Norway_", "Norway"),
)


def resolve_sample_country(
    sample: Union[str, Path, Dict[str, Any]],
    manifest_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve the canonical sovereign country for a sample.

    Resolution Strategy:
    1. If `manifest_lookup` is provided, look up by basename, stem, or path string.
    2. If `sample` is a dict (e.g., manifest row), check 'country', 'subset', or 'image_id'.
    3. If `sample` is a Path or str, match the filename stem against canonical prefixes.

    Args:
        sample: File path, filename, image ID, or record dictionary.
        manifest_lookup: Optional dictionary mapping image identifiers/stems to country.

    Returns:
        Canonical sovereign country name (one of LOCO_COUNTRIES).

    Raises:
        ValueError: If country cannot be resolved or is not in LOCO_COUNTRIES.
    """
    # 1. Manifest lookup
    if manifest_lookup is not None:
        if isinstance(sample, (str, Path)):
            path_obj = Path(sample)
            candidates = [
                str(sample),
                path_obj.name,
                path_obj.stem,
            ]
            for cand in candidates:
                if cand in manifest_lookup:
                    country = manifest_lookup[cand]
                    return _normalize_country(country)

    # 2. Dict record
    if isinstance(sample, dict):
        if "country" in sample and sample["country"]:
            return _normalize_country(str(sample["country"]))
        if "subset" in sample and sample["subset"]:
            return _normalize_country(str(sample["subset"]))
        if "image_id" in sample and sample["image_id"]:
            return resolve_sample_country(str(sample["image_id"]), manifest_lookup=None)
        if "image_path" in sample and sample["image_path"]:
            return resolve_sample_country(str(sample["image_path"]), manifest_lookup=None)
        raise ValueError(f"Record dict missing country/subset/image_id fields: {sample}")

    # 3. Path / string prefix matching
    stem = Path(sample).stem
    for prefix, country in COUNTRY_PREFIX_MAP:
        if stem.startswith(prefix):
            return country

    # Check exact country match if stem equals country name
    normalized_stem = _normalize_country(stem, strict=False)
    if normalized_stem in LOCO_COUNTRIES:
        return normalized_stem

    raise ValueError(
        f"Unable to resolve country for sample identifier: '{sample}'. "
        f"Filename stem must start with one of: {[p[0] for p in COUNTRY_PREFIX_MAP]}."
    )


def _normalize_country(country_str: str, strict: bool = True) -> str:
    """Normalize country or subset string into canonical sovereign country."""
    country_clean = country_str.strip()
    if country_clean in CHINA_SUBSETS or country_clean == "China":
        return "China"
    if country_clean in LOCO_COUNTRIES:
        return country_clean

    if strict:
        raise ValueError(
            f"Unrecognized country '{country_clean}'. Expected one of {LOCO_COUNTRIES} "
            f"or China subsets {CHINA_SUBSETS}."
        )
    return country_clean


def build_country_index_map(
    samples: Sequence[Any],
    target_country: Optional[str] = None,
    manifest_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """Index dataset samples by their sovereign source country.

    Filters out any samples belonging to `target_country` to guarantee strict
    zero target-domain leakage during source training.

    Args:
        samples: Sequence of sample paths, stems, or record dicts.
        target_country: Held-out country to strictly exclude from training.
        manifest_lookup: Optional lookup dictionary from fold manifest.

    Returns:
        Tuple of:
            - country_indices: Mapping from sovereign country to list of sample indices.
            - excluded_indices: Indices of samples excluded because they belong to target_country.
    """
    normalized_target = _normalize_country(target_country) if target_country else None

    country_indices: Dict[str, List[int]] = defaultdict(list)
    excluded_indices: List[int] = []

    for idx, sample in enumerate(samples):
        country = resolve_sample_country(sample, manifest_lookup=manifest_lookup)

        if normalized_target is not None and country == normalized_target:
            excluded_indices.append(idx)
        else:
            country_indices[country].append(idx)

    # Convert to standard dict with sorted country keys for determinism
    ordered_indices = {c: country_indices[c] for c in sorted(country_indices.keys())}
    return ordered_indices, excluded_indices


class CountryBalancedBatchSampler(Sampler[List[int]]):
    """Deterministic, Country-Balanced Batch Sampler for PyTorch / Ultralytics.

    Guarantees that each mini-batch (or sequence of mini-batches) is balanced
    across all active source countries, mitigating dominance by overrepresented
    countries (e.g. Japan) and underrepresentation of smaller countries (e.g. US).

    Key Features:
    - Zero Target Leakage: Strictly excludes any held-out country samples.
    - Unequal Country Sizes: Cycles through underrepresented countries with
      deterministic reshuffling (sampling with replacement where necessary).
    - Non-Divisible Batch Sizes: Evenly distributes floor(B / K) slots across
      all K active countries, and rotates the remainder across batches.
    - Determinism: Controlled by seed + epoch via `set_epoch(epoch)`.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        batch_size: int,
        target_country: Optional[str] = None,
        manifest_lookup: Optional[Dict[str, str]] = None,
        seed: int = 42,
        drop_last: bool = False,
        epoch_mode: str = "total",
        num_batches: Optional[int] = None,
    ) -> None:
        """Initialize the country-balanced batch sampler.

        Args:
            samples: List of image paths, filenames, or records.
            batch_size: Mini-batch size (e.g., 16).
            target_country: Held-out country to strictly isolate and exclude.
            manifest_lookup: Optional mapping from sample identifier to country.
            seed: Base random seed for deterministic sampling.
            drop_last: If True, drop incomplete batches if any occur.
            epoch_mode: Method for determining epoch size:
                - 'total': Total samples equal total source images.
                - 'max': Oversample so each country gets max(country_size) samples.
                - 'min': Undersample so each country gets min(country_size) samples.
            num_batches: Explicit total number of batches per epoch (overrides epoch_mode).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.batch_size = batch_size
        self.target_country = _normalize_country(target_country) if target_country else None
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.epoch_mode = epoch_mode

        # Partition dataset indices by country
        self.country_indices, self.excluded_indices = build_country_index_map(
            samples, target_country=self.target_country, manifest_lookup=manifest_lookup
        )

        self.active_countries: List[str] = sorted(self.country_indices.keys())
        if not self.active_countries:
            raise ValueError("No active source country samples found to construct sampler.")

        self.num_countries = len(self.active_countries)
        self.country_sizes = {c: len(self.country_indices[c]) for c in self.active_countries}
        self.total_source_samples = sum(self.country_sizes.values())

        # Determine number of batches per epoch
        if num_batches is not None:
            self.total_batches = num_batches
        elif epoch_mode == "total":
            # Match standard baseline epoch size
            self.total_batches = (
                self.total_source_samples // self.batch_size
                if drop_last
                else math.ceil(self.total_source_samples / self.batch_size)
            )
        elif epoch_mode == "max":
            max_size = max(self.country_sizes.values())
            total_samples = max_size * self.num_countries
            self.total_batches = (
                total_samples // self.batch_size if drop_last else math.ceil(total_samples / self.batch_size)
            )
        elif epoch_mode == "min":
            min_size = min(self.country_sizes.values())
            total_samples = min_size * self.num_countries
            self.total_batches = (
                total_samples // self.batch_size if drop_last else math.ceil(total_samples / self.batch_size)
            )
        else:
            raise ValueError(f"Unknown epoch_mode '{epoch_mode}'. Choose 'total', 'max', or 'min'.")

        self.total_batches = max(1, self.total_batches)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic sample shuffling."""
        self.epoch = epoch

    def _get_country_slots_for_batch(self, batch_idx: int) -> Dict[str, int]:
        """Calculate how many samples each country should contribute to this batch."""
        base_quota = self.batch_size // self.num_countries
        remainder = self.batch_size % self.num_countries

        slots: Dict[str, int] = {c: base_quota for c in self.active_countries}

        # Distribute remainder slots round-robin across batches
        for r in range(remainder):
            lucky_country_idx = (batch_idx * remainder + r) % self.num_countries
            lucky_country = self.active_countries[lucky_country_idx]
            slots[lucky_country] += 1

        return slots

    def __iter__(self) -> Iterator[List[int]]:
        """Yield balanced batches of dataset indices."""
        # Seeded deterministic RNG per country
        rng = random.Random(self.seed + self.epoch * 10007)

        # Build shuffled pools for each country
        shuffled_pools: Dict[str, List[int]] = {}
        pool_ptrs: Dict[str, int] = {}

        for country, indices in self.country_indices.items():
            pool = list(indices)
            rng.shuffle(pool)
            shuffled_pools[country] = pool
            pool_ptrs[country] = 0

        for b_idx in range(self.total_batches):
            slots = self._get_country_slots_for_batch(b_idx)
            batch: List[int] = []

            for country in self.active_countries:
                needed = slots[country]
                pool = shuffled_pools[country]
                ptr = pool_ptrs[country]

                while needed > 0:
                    available = len(pool) - ptr
                    if available >= needed:
                        batch.extend(pool[ptr : ptr + needed])
                        pool_ptrs[country] = ptr + needed
                        needed = 0
                    else:
                        # Take remaining and reshuffle with replacement
                        batch.extend(pool[ptr:])
                        needed -= available
                        rng.shuffle(pool)
                        ptr = 0
                        pool_ptrs[country] = 0

            # Shuffle within batch so countries are not strictly ordered
            rng.shuffle(batch)

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.total_batches


class CountryBalancedSampler(Sampler[int]):
    """Deterministic, Country-Balanced Index Sampler (Item-Level) for PyTorch.

    Yields individual sample indices such that all active source countries
    are drawn with uniform probability throughout the epoch.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        target_country: Optional[str] = None,
        manifest_lookup: Optional[Dict[str, str]] = None,
        seed: int = 42,
        samples_per_epoch: Optional[int] = None,
        interleave: bool = True,
    ) -> None:
        """Initialize item-level country-balanced sampler."""
        self.target_country = _normalize_country(target_country) if target_country else None
        self.seed = seed
        self.epoch = 0
        self.interleave = interleave

        self.country_indices, self.excluded_indices = build_country_index_map(
            samples, target_country=self.target_country, manifest_lookup=manifest_lookup
        )

        self.active_countries: List[str] = sorted(self.country_indices.keys())
        if not self.active_countries:
            raise ValueError("No active source country samples found to construct sampler.")

        self.num_countries = len(self.active_countries)
        self.country_sizes = {c: len(self.country_indices[c]) for c in self.active_countries}
        self.total_source_samples = sum(self.country_sizes.values())

        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            self.samples_per_epoch = self.total_source_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield balanced sample indices."""
        rng = random.Random(self.seed + self.epoch * 10007)

        # Quotas per country
        quota_per_country = self.samples_per_epoch // self.num_countries
        remainder = self.samples_per_epoch % self.num_countries

        country_quotas = {c: quota_per_country for c in self.active_countries}
        for r in range(remainder):
            lucky_country = self.active_countries[(self.epoch + r) % self.num_countries]
            country_quotas[lucky_country] += 1

        # Sample for each country
        selected_per_country: Dict[str, List[int]] = {}
        for country in self.active_countries:
            quota = country_quotas[country]
            indices = list(self.country_indices[country])
            rng.shuffle(indices)

            # Cycle with replacement if quota exceeds country sample count
            selected: List[int] = []
            while len(selected) < quota:
                needed = quota - len(selected)
                if len(indices) >= needed:
                    selected.extend(indices[:needed])
                else:
                    selected.extend(indices)
                    rng.shuffle(indices)

            selected_per_country[country] = selected

        if self.interleave:
            # Interleave indices across countries for uniform distribution
            interleaved: List[int] = []
            max_quota = max(country_quotas.values())
            for i in range(max_quota):
                for country in self.active_countries:
                    if i < len(selected_per_country[country]):
                        interleaved.append(selected_per_country[country][i])
            return iter(interleaved)
        else:
            flat_indices: List[int] = []
            for country in self.active_countries:
                flat_indices.extend(selected_per_country[country])
            rng.shuffle(flat_indices)
            return iter(flat_indices)

    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.samples_per_epoch


def load_manifest_country_lookup(manifest_path_or_df: Union[str, Path, pd.DataFrame]) -> Dict[str, str]:
    """Build a fast sample-to-country lookup dictionary from a fold manifest.

    Args:
        manifest_path_or_df: Path to fold_manifest.csv or loaded DataFrame.

    Returns:
        Dictionary mapping image_id, image_name, and image_path to normalized country.
    """
    if isinstance(manifest_path_or_df, (str, Path)):
        p = Path(manifest_path_or_df)
        if not p.is_file():
            raise FileNotFoundError(f"Manifest file not found: {p}")
        df = pd.read_csv(p)
    else:
        df = manifest_path_or_df

    lookup: Dict[str, str] = {}
    for _, row in df.iterrows():
        country = _normalize_country(str(row.get("country", row.get("subset", ""))))

        if "image_id" in row and pd.notna(row["image_id"]):
            img_id = str(row["image_id"]).strip()
            lookup[img_id] = country
            lookup[f"{img_id}.jpg"] = country

        if "image_path" in row and pd.notna(row["image_path"]):
            p_str = str(row["image_path"]).strip()
            lookup[p_str] = country
            lookup[Path(p_str).name] = country
            lookup[Path(p_str).stem] = country

    return lookup


def create_country_balanced_batch_sampler(
    dataset: Any,
    batch_size: int,
    target_country: Optional[str] = None,
    manifest_path_or_df: Optional[Union[str, Path, pd.DataFrame]] = None,
    seed: int = 42,
    epoch: int = 0,
    drop_last: bool = False,
    epoch_mode: str = "total",
) -> CountryBalancedBatchSampler:
    """Convenience factory to create a CountryBalancedBatchSampler from a dataset.

    Compatible with Ultralytics YOLODataset (which exposes .im_files) or standard
    PyTorch datasets exposing a list of sample paths.

    Args:
        dataset: Dataset instance or list of image paths/records.
        batch_size: Mini-batch size.
        target_country: Held-out country to isolate.
        manifest_path_or_df: Optional fold_manifest.csv or DataFrame.
        seed: Base random seed.
        epoch: Initial epoch index.
        drop_last: Whether to drop the last incomplete batch.
        epoch_mode: Epoch sizing method ('total', 'max', 'min').

    Returns:
        Configured CountryBalancedBatchSampler ready for torch.utils.data.DataLoader.
    """
    # Extract samples list
    if hasattr(dataset, "im_files"):
        samples = dataset.im_files
    elif isinstance(dataset, (list, tuple)):
        samples = dataset
    else:
        # Fallback to indexing dataset by len
        samples = [dataset[i] for i in range(len(dataset))]

    manifest_lookup = (
        load_manifest_country_lookup(manifest_path_or_df)
        if manifest_path_or_df is not None
        else None
    )

    sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=batch_size,
        target_country=target_country,
        manifest_lookup=manifest_lookup,
        seed=seed,
        drop_last=drop_last,
        epoch_mode=epoch_mode,
    )
    sampler.set_epoch(epoch)
    return sampler


class CBSSamplerProxy(Sampler[int]):
    """Proxy sampler exposed on train_loader.sampler for Ultralytics lifecycle compatibility.

    Ultralytics 8.4+ (_build_train_pipeline and DDP) queries:
    1. len(train_loader.sampler) to calculate epoch sample count.
    2. train_loader.sampler.set_epoch(epoch) during DDP epoch start.

    This proxy ensures that both operations work seamlessly when CountryBalancedBatchSampler
    is used as the underlying batch sampler.
    """

    def __init__(self, cbs_sampler: Any, total_samples: int) -> None:
        self.cbs_sampler = cbs_sampler
        self.total_samples = total_samples

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self) -> Iterator[int]:
        return iter(self.cbs_sampler)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.cbs_sampler, "set_epoch"):
            self.cbs_sampler.set_epoch(epoch)


class CCARCInfiniteDataLoader(InfiniteDataLoader):
    """InfiniteDataLoader subclass preserving integer batch_size and CBS sampler for Ultralytics.

    PyTorch's DataLoader.__init__ unconditionally sets self.batch_size = None when a
    batch_sampler is provided. Ultralytics 8.4+ (_build_train_pipeline) evaluates:
        final_batch_size = len(self.train_loader.sampler) % self.train_loader.batch_size or self.train_loader.batch_size

    This subclass preserves self.batch_size as the genuine integer batch_size and
    attaches CBSSamplerProxy to self.sampler, ensuring full compatibility with Ultralytics 8.4+.
    """

    def __init__(
        self,
        *args: Any,
        batch_size: int = 16,
        cbs_sampler: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_size", int(batch_size))
        if cbs_sampler is not None:
            total_samples = getattr(
                cbs_sampler, "total_source_samples", len(getattr(self, "dataset", []))
            )
            proxy = CBSSamplerProxy(cbs_sampler, total_samples)
            object.__setattr__(self, "sampler", proxy)

