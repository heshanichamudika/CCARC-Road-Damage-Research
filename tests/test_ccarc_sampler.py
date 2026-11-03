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
    CBSSamplerProxy,
    CCARCInfiniteDataLoader,
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


def test_batch_size_none_regression_and_ultralytics_dataloader_fix():
    """Reproduce the original batch_size=None TypeError and verify CCARCInfiniteDataLoader fix."""
    class SimpleDataset(Dataset):
        def __init__(self, n: int = 20):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return torch.tensor([idx])

    dataset = SimpleDataset(20)
    samples = [f"Japan_{i:02d}.jpg" for i in range(10)] + [f"Czech_{i:02d}.jpg" for i in range(10)]
    batch_sampler = CountryBalancedBatchSampler(samples=samples, batch_size=4, seed=42)

    # 1. Reproduce the failure: standard PyTorch DataLoader with batch_sampler sets batch_size=None
    standard_loader = DataLoader(dataset, batch_sampler=batch_sampler)
    assert standard_loader.batch_size is None

    # This exact expression from Ultralytics 8.4+ _build_train_pipeline raises TypeError
    with pytest.raises(TypeError, match="unsupported operand type\\(s\\) for %: 'int' and 'NoneType'"):
        _ = len(standard_loader.sampler) % standard_loader.batch_size

    # 2. Verify CCARCInfiniteDataLoader sets valid integer batch_size and integrates CBSSamplerProxy
    cbs_loader = CCARCInfiniteDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        batch_size=4,
        cbs_sampler=batch_sampler,
        num_workers=0,
    )
    assert isinstance(cbs_loader.batch_size, int)
    assert cbs_loader.batch_size == 4

    # Ultralytics 8.4+ expression now evaluates cleanly without error
    final_batch_size = len(cbs_loader.sampler) % cbs_loader.batch_size or cbs_loader.batch_size
    assert isinstance(final_batch_size, int)
    assert final_batch_size > 0

    # Verify InfiniteDataLoader features
    assert hasattr(cbs_loader, "reset")
    first_batch = next(iter(cbs_loader))
    assert first_batch.shape[0] == 4


def test_cbs_norway_exclusion_and_five_source_countries_representation():
    """Verify that Fold 4 excludes Norway and balances across all 5 source countries."""
    samples = [
        "China_Drone_01.jpg", "China_MotorBike_02.jpg",
        "Czech_01.jpg", "Czech_02.jpg",
        "India_01.jpg", "India_02.jpg",
        "Japan_01.jpg", "Japan_02.jpg",
        "United_States_01.jpg", "United_States_02.jpg",
        "Norway_01.jpg", "Norway_02.jpg",  # Target country to isolate
    ]

    class NamedDataset(Dataset):
        def __init__(self, files):
            self.files = files
        def __len__(self):
            return len(self.files)
        def __getitem__(self, idx):
            return resolve_sample_country(self.files[idx])

    dataset = NamedDataset(samples)
    batch_size = 5  # 1 sample per source country

    batch_sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=batch_size,
        target_country="Norway",
        seed=42,
    )

    # 1. Verify country partitioning
    expected_active = ["China", "Czech", "India", "Japan", "United_States"]
    assert sorted(batch_sampler.active_countries) == expected_active
    assert len(batch_sampler.excluded_indices) == 2  # Both Norway samples excluded

    loader = CCARCInfiniteDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        batch_size=batch_size,
        cbs_sampler=batch_sampler,
        num_workers=0,
    )

    # 2. Iterate and verify batches
    country_totals = Counter()
    for batch_countries in loader:
        # Strict zero-leakage check
        assert "Norway" not in batch_countries
        assert len(batch_countries) == 5
        # Verify representation in the batch
        batch_set = set(batch_countries)
        assert len(batch_set) == 5, f"Batch missing country representation: {batch_countries}"
        country_totals.update(batch_countries)

    # All 5 countries should have equal representation across the epoch
    assert len(country_totals) == 5
    counts = list(country_totals.values())
    assert all(c == counts[0] for c in counts)


def test_cbs_sampler_proxy_epoch_synchronization():
    """Verify that CBSSamplerProxy propagates set_epoch to the underlying CBS sampler."""
    samples = ["Japan_01.jpg", "Czech_01.jpg", "India_01.jpg"]
    batch_sampler = CountryBalancedBatchSampler(samples=samples, batch_size=2, seed=42)

    proxy = CBSSamplerProxy(batch_sampler, total_samples=len(samples))
    assert hasattr(proxy, "set_epoch")

    # Set epoch and verify state change
    proxy.set_epoch(5)
    assert batch_sampler.epoch == 5
    assert len(proxy) == len(samples)


def test_ccarc_infinite_dataloader_reset_mosaic_update_and_cbs_preservation():
    """Verify CCARCInfiniteDataLoader.reset() works, updates dynamic dataset transforms, and preserves CBS."""
    class DynamicTransformDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
            self.mosaic = True

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            country = resolve_sample_country(self.samples[idx])
            return {
                "idx": idx,
                "country": country,
                "is_mosaic": self.mosaic,
            }

        def close_mosaic(self):
            self.mosaic = False

    samples = [
        "China_01.jpg", "Czech_01.jpg", "India_01.jpg", "Japan_01.jpg", "United_States_01.jpg",
        "China_02.jpg", "Czech_02.jpg", "India_02.jpg", "Japan_02.jpg", "United_States_02.jpg",
    ]
    dataset = DynamicTransformDataset(samples)
    batch_size = 5
    batch_sampler = CountryBalancedBatchSampler(
        samples=samples,
        batch_size=batch_size,
        target_country="Norway",
        seed=42,
    )

    loader = CCARCInfiniteDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        batch_size=batch_size,
        cbs_sampler=batch_sampler,
        num_workers=0,
    )

    # 1. Verify attributes and reset method existence
    assert hasattr(loader, "reset")
    assert callable(loader.reset)
    assert isinstance(loader.batch_size, int)
    assert loader.batch_size == 5

    # 2. Iterate first epoch: mosaic is True
    first_epoch_batches = list(loader)
    assert len(first_epoch_batches) == len(batch_sampler)
    for b in first_epoch_batches:
        assert all(b["is_mosaic"])
        # Check country representation
        assert set(b["country"]) == {"China", "Czech", "India", "Japan", "United_States"}

    # 3. Simulate Ultralytics mosaic closing and call loader.reset()
    dataset.close_mosaic()
    loader.reset()

    # 4. Verify properties after reset
    assert isinstance(loader.batch_size, int)
    assert loader.batch_size == 5
    assert hasattr(loader.sampler, "set_epoch")

    # 5. Verify that second epoch receives updated transforms (is_mosaic is False) and preserves CBS
    second_epoch_batches = list(loader)
    assert len(second_epoch_batches) == len(batch_sampler)
    for b in second_epoch_batches:
        assert not any(b["is_mosaic"])
        assert set(b["country"]) == {"China", "Czech", "India", "Japan", "United_States"}

    # 6. Verify epoch synchronization
    loader.sampler.set_epoch(10)
    assert batch_sampler.epoch == 10


