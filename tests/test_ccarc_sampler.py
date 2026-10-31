"""Unit Tests for Country-Balanced Sampling (CBS) in CCARC.

Verifies country resolution, China subset unification, strict held-out target
isolation, deterministic epoch-level sampling, unequal country balancing,
non-divisible batch handling, and PyTorch / Ultralytics compatibility interfaces.
"""

from collections import Counter
from pathlib import Path
import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.loco_splitter import CHINA_SUBSETS, LOCO_COUNTRIES
from src.training.ccarc_sampler import (
    CountryBalancedBatchSampler,
    CountryBalancedSampler,
    build_country_index_map,
    create_country_balanced_batch_sampler,
    load_manifest_country_lookup,
    resolve_sample_country,
)


# Helper to generate a realistic mock source/target dataset
def create_mock_dataset_paths() -> list[str]:
    """Create sample paths representing imbalanced country distributions."""
    samples = []
    # Japan (large: 100 samples)
    for i in range(100):
        samples.append(f"/datasets/loco_views/fold_0/images/train/Japan_{i:04d}.jpg")
    # Czech (small: 20 samples)
    for i in range(20):
        samples.append(f"/datasets/loco_views/fold_0/images/train/Czech_{i:04d}.jpg")
    # India (medium: 50 samples)
    for i in range(50):
        samples.append(f"/datasets/loco_views/fold_0/images/train/India_{i:04d}.jpg")
    # Norway (medium: 40 samples)
    for i in range(40):
        samples.append(f"/datasets/loco_views/fold_0/images/train/Norway_{i:04d}.jpg")
    # United_States (small: 25 samples)
    for i in range(25):
        samples.append(f"/datasets/loco_views/fold_0/images/train/United_States_{i:04d}.jpg")
    # China subsets (if included in source: Drone 15, MotorBike 15 -> total 30)
    for i in range(15):
        samples.append(f"/datasets/loco_views/fold_0/images/train/China_Drone_{i:04d}.jpg")
    for i in range(15):
        samples.append(f"/datasets/loco_views/fold_0/images/train/China_MotorBike_{i:04d}.jpg")
    return samples


def test_resolve_sample_country_prefixes():
    """Verify prefix-based resolution across all six countries and China subsets."""
    assert resolve_sample_country("Japan_001265.jpg") == "Japan"
    assert resolve_sample_country("India_000042.jpg") == "India"
    assert resolve_sample_country("Czech_000100.jpg") == "Czech"
    assert resolve_sample_country("Norway_003412.jpg") == "Norway"
    assert resolve_sample_country("United_States_000001.jpg") == "United_States"
    assert resolve_sample_country("China_Drone_000050.jpg") == "China"
    assert resolve_sample_country("China_MotorBike_000099.jpg") == "China"
    assert resolve_sample_country("China_000012.jpg") == "China"

    # Paths with directory structure
    assert resolve_sample_country(Path("/path/to/images/train/Japan_001265.jpg")) == "Japan"
    assert resolve_sample_country(Path("C:/datasets/Norway_003412.png")) == "Norway"


def test_resolve_sample_country_manifest_dict():
    """Verify resolving country from manifest row dictionary."""
    row1 = {"country": "Japan", "image_id": "Japan_001"}
    assert resolve_sample_country(row1) == "Japan"

    row2 = {"subset": "China_Drone", "image_id": "China_Drone_001"}
    assert resolve_sample_country(row2) == "China"

    row3 = {"subset": "China_MotorBike", "image_id": "China_MotorBike_001"}
    assert resolve_sample_country(row3) == "China"

    row4 = {"country": "United_States", "image_id": "US_001"}
    assert resolve_sample_country(row4) == "United_States"


def test_resolve_sample_country_manifest_lookup(tmp_path: Path):
    """Verify resolving country via loaded manifest CSV lookup table."""
    manifest_file = tmp_path / "fold_manifest.csv"
    df = pd.DataFrame([
        {"image_id": "cust_001", "country": "Japan", "image_path": "/data/cust_001.jpg"},
        {"image_id": "cust_002", "country": "China", "subset": "China_Drone", "image_path": "/data/cust_002.jpg"},
        {"image_id": "cust_003", "country": "United_States", "image_path": "/data/cust_003.jpg"},
    ])
    df.to_csv(manifest_file, index=False)

    lookup = load_manifest_country_lookup(manifest_file)
    assert resolve_sample_country("cust_001.jpg", manifest_lookup=lookup) == "Japan"
    assert resolve_sample_country("cust_002.jpg", manifest_lookup=lookup) == "China"
    assert resolve_sample_country("cust_003.jpg", manifest_lookup=lookup) == "United_States"


def test_resolve_sample_country_invalid_raises():
    """Verify invalid sample stems raise ValueError."""
    with pytest.raises(ValueError, match="Unable to resolve country"):
        resolve_sample_country("UnknownCountry_0001.jpg")

    with pytest.raises(ValueError, match="Unable to resolve country"):
        resolve_sample_country("image_without_prefix.jpg")


def test_build_country_index_map_and_china_unification():
    """Verify mapping builds correct country pools and unifies China drone/motorbike."""
    samples = [
        "Japan_001.jpg",
        "Japan_002.jpg",
        "China_Drone_001.jpg",
        "China_MotorBike_001.jpg",
        "India_001.jpg",
    ]
    mapping, excluded = build_country_index_map(samples)

    assert set(mapping.keys()) == {"Japan", "China", "India"}
    assert mapping["Japan"] == [0, 1]
    assert mapping["China"] == [2, 3]  # Both China_Drone and China_MotorBike in China
    assert mapping["India"] == [4]
    assert len(excluded) == 0


def test_held_out_country_isolation():
    """Verify that held-out target country images are strictly excluded from the sampling pools."""
    samples = create_mock_dataset_paths()  # Contains China, Japan, Czech, India, Norway, United_States
    target_country = "China"

    sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=16,
        target_country=target_country,
        seed=42,
    )

    # China must be in excluded_indices and NOT in active_countries
    assert "China" not in sampler.active_countries
    assert len(sampler.excluded_indices) == 30  # 15 Drone + 15 MotorBike

    # All generated batches must NEVER contain any sample from China
    all_sampled_indices = []
    for batch in sampler:
        for idx in batch:
            all_sampled_indices.append(idx)
            sample_country = resolve_sample_country(samples[idx])
            assert sample_country != "China", f"Leakage detected: sampled {samples[idx]} from target country!"

    assert not any(idx in sampler.excluded_indices for idx in all_sampled_indices)


def test_batch_sampler_country_balance_across_batches():
    """Verify that batches achieve balanced representation across active countries."""
    samples = create_mock_dataset_paths()
    # Hold out China; remaining 5 source countries: Czech, India, Japan, Norway, United_States
    batch_size = 15  # Divisible by 5: exactly 3 per country per batch
    sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=batch_size,
        target_country="China",
        seed=42,
        epoch_mode="total",
    )

    assert len(sampler.active_countries) == 5
    for batch in sampler:
        assert len(batch) == batch_size
        countries_in_batch = [resolve_sample_country(samples[idx]) for idx in batch]
        counts = Counter(countries_in_batch)
        for country in sampler.active_countries:
            assert counts[country] == 3, f"Imbalance in batch: {counts}"


def test_batch_sampler_non_divisible_batch_size():
    """Verify handling non-divisible batch size (e.g. batch_size=16 across 5 countries)."""
    samples = create_mock_dataset_paths()
    batch_size = 16  # 16 = 5 * 3 + 1 remainder
    sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=batch_size,
        target_country="China",
        seed=42,
        epoch_mode="total",
    )

    total_counts = Counter()
    for batch in sampler:
        assert len(batch) == batch_size
        countries_in_batch = [resolve_sample_country(samples[idx]) for idx in batch]
        counts = Counter(countries_in_batch)

        # Every country should have at least floor(16 / 5) = 3 samples, and at most 4
        for country in sampler.active_countries:
            assert 3 <= counts[country] <= 4
        total_counts.update(countries_in_batch)

    # Across all batches in the epoch, countries should be near-identically balanced
    max_count = max(total_counts.values())
    min_count = min(total_counts.values())
    assert (max_count - min_count) <= len(sampler)  # Bound remainder drift


def test_small_batch_size_edge_case():
    """Verify handling when batch size is smaller than number of countries (e.g. batch_size=2 with 5 countries)."""
    samples = create_mock_dataset_paths()
    batch_size = 2
    sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=batch_size,
        target_country="China",
        seed=42,
        epoch_mode="total",
    )

    all_sampled = []
    for batch in sampler:
        assert len(batch) == batch_size
        all_sampled.extend([resolve_sample_country(samples[idx]) for idx in batch])

    counts = Counter(all_sampled)
    # Over the entire epoch, round-robin ensures close balance
    assert max(counts.values()) - min(counts.values()) <= 2


def test_deterministic_epoch_sampling():
    """Verify reproducibility: same epoch yields identical order, different epoch yields different order."""
    samples = create_mock_dataset_paths()
    sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=16,
        target_country="Norway",
        seed=42,
    )

    sampler.set_epoch(0)
    epoch_0_run_1 = list(sampler)

    sampler.set_epoch(0)
    epoch_0_run_2 = list(sampler)

    assert epoch_0_run_1 == epoch_0_run_2, "Sampling at same epoch must be strictly deterministic!"

    sampler.set_epoch(1)
    epoch_1_run = list(sampler)
    assert epoch_0_run_1 != epoch_1_run, "Sampling at different epoch must produce different permutations!"


def test_item_level_country_balanced_sampler():
    """Verify item-level CountryBalancedSampler yields uniform country distribution."""
    samples = create_mock_dataset_paths()
    # 5 source countries (excluding United_States)
    sampler = CountryBalancedSampler(
        samples=samples,
        target_country="United_States",
        seed=42,
        samples_per_epoch=200,
        interleave=True,
    )

    assert len(sampler) == 200
    indices = list(sampler)
    assert len(indices) == 200

    sampled_countries = [resolve_sample_country(samples[idx]) for idx in indices]
    counts = Counter(sampled_countries)

    # 200 / 5 = 40 samples per country
    assert len(counts) == 5
    for country in sampler.active_countries:
        assert counts[country] == 40
        assert country != "United_States"


def test_pytorch_dataloader_integration():
    """Verify compatibility with torch.utils.data.DataLoader."""
    samples = create_mock_dataset_paths()

    class MockRDDDataset(Dataset):
        def __init__(self, paths):
            self.im_files = paths

        def __len__(self):
            return len(self.im_files)

        def __getitem__(self, idx):
            # Return dummy tensor and country label
            return torch.zeros((3, 32, 32)), resolve_sample_country(self.im_files[idx])

    dataset = MockRDDDataset(samples)
    batch_sampler = create_country_balanced_batch_sampler(
        dataset=dataset,
        batch_size=16,
        target_country="Japan",
        seed=42,
    )

    loader = DataLoader(dataset, batch_sampler=batch_sampler)

    batches_seen = 0
    for imgs, country_names in loader:
        batches_seen += 1
        assert imgs.shape[0] == 16
        # Confirm Japan is never present in loaded batches
        assert "Japan" not in country_names

    assert batches_seen == len(batch_sampler)


def test_unequal_country_sizes_oversampling():
    """Verify that smaller countries (e.g. 20 samples) safely cycle without error or starvation."""
    # Japan has 100 samples, Czech has only 5 samples
    samples = [f"Japan_{i:04d}.jpg" for i in range(100)] + [f"Czech_{i:04d}.jpg" for i in range(5)]

    sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=10,  # 5 Japan, 5 Czech per batch
        seed=42,
        epoch_mode="total",
    )

    czech_seen = 0
    japan_seen = 0
    for batch in sampler:
        for idx in batch:
            c = resolve_sample_country(samples[idx])
            if c == "Czech":
                czech_seen += 1
            elif c == "Japan":
                japan_seen += 1

    # Over the epoch, both countries should have had equal opportunity despite 100 vs 5 raw count
    assert czech_seen > 0
    assert japan_seen > 0
    assert abs(czech_seen - japan_seen) <= len(sampler)
