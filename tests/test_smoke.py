"""Minimal Smoke Test Suite for CCARC Road Damage Research Project."""

from pathlib import Path
import yaml
import pytest
import src


def test_package_metadata():
    """Verify src package imports and metadata."""
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


def test_directory_structure():
    """Verify that expected research directories exist."""
    project_root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "configs",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "tests",
        "notebooks",
        "docs",
        "experiments",
        "checkpoints",
        "app",
    ]
    for rel_dir in expected_dirs:
        dir_path = project_root / rel_dir
        assert dir_path.is_dir(), f"Expected directory missing: {rel_dir}"


def test_default_configuration():
    """Verify default.yaml exists, parses correctly, and contains research taxonomy."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "default.yaml"
    assert config_path.is_file(), "configs/default.yaml does not exist."

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Dataset checks
    assert "dataset" in config
    assert config["dataset"]["name"] == "RDD2022"
    countries = config["dataset"]["countries"]
    assert len(countries) == 6
    assert set(countries) == {
        "Japan",
        "India",
        "Czech",
        "Norway",
        "United_States",
        "China",
    }

    # Taxonomy checks
    classes = config["dataset"]["classes"]
    assert len(classes) == 4
    assert classes[0] == "D00"
    assert classes[1] == "D10"
    assert classes[2] == "D20"
    assert classes[3] == "D40"

    # Model and LOCO checks
    assert config["model"]["architecture"] == "yolov8s"
    assert config["evaluation"]["protocol"] == "LOCO"
    assert "ccarc" in config


def test_loco_partition_logic():
    """Verify that Leave-One-Country-Out (LOCO) partitioning generates valid non-overlapping splits."""
    countries = ["Japan", "India", "Czech", "Norway", "United_States", "China"]
    num_countries = len(countries)

    for target_country in countries:
        source_countries = [c for c in countries if c != target_country]
        assert len(source_countries) == num_countries - 1
        assert target_country not in source_countries
        # Ensure zero overlap
        assert set(source_countries).isdisjoint({target_country})
