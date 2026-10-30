#!/usr/bin/env python3
"""Environment Verification Script for CCARC Road Damage Research Project.

Checks Python version, essential package installations, hardware/CUDA
availability, and directory structure integrity.
"""

import importlib
import os
import platform
import sys
from pathlib import Path

REQUIRED_DIRECTORIES = [
    "configs",
    "src/data",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/utils",
    "tests",
    "notebooks",
    "docs",
    "experiments/runs",
    "experiments/logs",
    "checkpoints",
    "app",
]

REQUIRED_PACKAGES = [
    ("torch", "PyTorch (Deep Learning Core)"),
    ("torchvision", "TorchVision (Vision Transforms & Models)"),
    ("ultralytics", "Ultralytics YOLOv8 (Detector Framework)"),
    ("cv2", "OpenCV (Image Processing)"),
    ("numpy", "NumPy (Numerical Computing)"),
    ("pandas", "Pandas (Tabular Data & Metrics)"),
    ("matplotlib", "Matplotlib (Plotting & Curves)"),
    ("sklearn", "Scikit-Learn (Metrics & Evaluation)"),
    ("scipy", "SciPy (Scientific Algorithms)"),
    ("yaml", "PyYAML (Configuration Parsing)"),
    ("pytest", "PyTest (Automated Testing)"),
    ("streamlit", "Streamlit (Research Dashboard)"),
]


def check_python_version() -> bool:
    print("=" * 70)
    print("1. PYTHON ENVIRONMENT")
    print("=" * 70)
    print(f"Executable:      {sys.executable}")
    print(f"Python Version:  {platform.python_version()}")
    print(f"Platform:        {platform.platform()}")
    print(f"Architecture:    {platform.architecture()[0]}")

    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 10:
        print("[PASS] Python version meets requirements (>= 3.10).")
        return True
    else:
        print(f"[FAIL] Python >= 3.10 is required. Found: {sys.version_info}")
        return False


def check_directory_structure(root_dir: Path) -> bool:
    print("\n" + "=" * 70)
    print("2. PROJECT DIRECTORY STRUCTURE")
    print("=" * 70)
    all_ok = True
    for relative_path in REQUIRED_DIRECTORIES:
        dir_path = root_dir / relative_path
        if dir_path.is_dir():
            print(f"  [OK]   {relative_path}/")
        else:
            print(f"  [MISS] {relative_path}/ (Missing)")
            all_ok = False

    if all_ok:
        print("[PASS] All required directories exist.")
    else:
        print("[FAIL] One or more directories are missing.")
    return all_ok


def check_packages() -> dict:
    print("\n" + "=" * 70)
    print("3. DEPENDENCY VERIFICATION")
    print("=" * 70)
    results = {}
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(
                mod, "__version__", getattr(mod, "VERSION", "Installed")
            )
            print(f"  [PASS] {display_name:<44} : {version}")
            results[module_name] = {"status": "PASS", "version": str(version)}
        except ImportError:
            print(f"  [FAIL] {display_name:<44} : NOT INSTALLED")
            results[module_name] = {"status": "FAIL", "version": None}

    return results


def check_cuda_device() -> None:
    print("\n" + "=" * 70)
    print("4. COMPUTE / ACCELERATOR CHECK")
    print("=" * 70)
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available via PyTorch: {cuda_available}")
        if cuda_available:
            print(f"Device Count:               {torch.cuda.device_count()}")
            print(f"Device Name:                {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version (PyTorch):     {torch.version.cuda}")
        else:
            print("Note: Running on CPU. (CUDA recommended for training YOLOv8s).")
    except ImportError:
        print(
            "PyTorch not yet installed; cannot probe CUDA capability directly."
        )


def main() -> int:
    root_dir = Path(__file__).resolve().parent

    py_ok = check_python_version()
    dir_ok = check_directory_structure(root_dir)
    pkg_results = check_packages()
    check_cuda_device()

    missing_pkgs = [
        name for name, info in pkg_results.items() if info["status"] == "FAIL"
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Python Check:      {'PASS' if py_ok else 'FAIL'}")
    print(f"Directory Check:   {'PASS' if dir_ok else 'FAIL'}")
    print(
        f"Packages Verified: {len(REQUIRED_PACKAGES) - len(missing_pkgs)}/{len(REQUIRED_PACKAGES)}"
    )

    if missing_pkgs:
        print(f"\nRemaining package blockers to install: {missing_pkgs}")
        print("To install required packages into your virtual environment:")
        print("  .\\.venv\\Scripts\\pip.exe install -r requirements.txt")
        return 1
    else:
        print("\nAll dependencies verified successfully. Environment is ready.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
