#!/usr/bin/env python3
"""CCARC Ablation Experiment Orchestrator and Checkpoint Safety Controller.

Provides safe, reproducible experiment orchestration for CCARC ablation studies (E0–E4)
across the six RDD2022 Leave-One-Country-Out (LOCO) folds.

Ablation Methodology (Orthogonal Single-Component Ablation):
- E0 = Standard YOLOv8s baseline (no CBS, no CAL, no RWC)
- E1 = Country-Balanced Sampling only (CBS only: cbs=True, cal=False, rwc=False)
- E2 = Class-Aware Loss only (CAL only: cbs=False, cal=True, rwc=False)
- E3 = Reliability-Weighted Consistency only (RWC only: cbs=False, cal=False, rwc=True)
- E4 = Full CCARC System (cbs=True, cal=True, rwc=True)

Key Capabilities:
1. Status Detection: Detects COMPLETED, IN_PROGRESS, PENDING, or SMOKE runs.
2. Interruption vs Completion: Accurately differentiates interrupted runs from completed or early-stopped runs.
3. Checkpoint Protection: Strictly refuses to overwrite completed runs or unconfirmed active runs.
4. Safe Resume: Verifies last.pt existence before emitting or executing `--resume`.
5. Exact CLI Command Generation: Generates copy-pasteable, reproducible commands with identical comparable hyperparameters.
6. Execution Control: Safe dry-run by default; optional sequential execution on explicit demand.
7. Structured Reports: Outputs JSON status dictionaries and formatted Markdown summaries without starting training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.training.ccarc_config import VALID_VARIANTS

# Standard Google Colab paths
COLAB_REPO_DIR = Path("/content/CCARC-Road-Damage-Research")
COLAB_VIEWS_DIR = Path("/content/rdd2022_recovery/loco_views")
COLAB_BASELINE_DIR = Path(
    "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolov8s_baseline_full100"
)
COLAB_CCARC_DIR = Path(
    "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/ccarc_full_training"
)
COLAB_LOCAL_RUN_DIR = Path("/content/runs/ccarc")

# Default local paths
DEFAULT_LOCAL_VIEWS_DIR = Path("experiments/loco_splits")
DEFAULT_LOCAL_BASELINE_DIR = Path("experiments/yolov8s_baseline_full100")
DEFAULT_LOCAL_CCARC_DIR = Path("experiments/ccarc_full_training")

# Variant component specification as implemented in src/training/ccarc_config.py
VARIANT_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "E0": {
        "name": "E0",
        "description": "Standard YOLOv8s Baseline (No CBS, No CAL, No RWC)",
        "cbs_enabled": False,
        "cal_enabled": False,
        "rwc_enabled": False,
        "architecture_type": "baseline",
        "module": "src.training.train_loco_baseline",
    },
    "E1": {
        "name": "E1",
        "description": "Country-Balanced Sampling (CBS only)",
        "cbs_enabled": True,
        "cal_enabled": False,
        "rwc_enabled": False,
        "architecture_type": "ablation_single",
        "module": "src.training.train_loco_ccarc",
    },
    "E2": {
        "name": "E2",
        "description": "Class-Aware Loss (CAL only in codebase implementation)",
        "cbs_enabled": False,
        "cal_enabled": True,
        "rwc_enabled": False,
        "architecture_type": "ablation_single",
        "module": "src.training.train_loco_ccarc",
        "discrepancy_note": (
            "Codebase implements E2 as CAL-only (cbs=False, cal=True, rwc=False). "
            "Literature/prompt variant outline listed E2 as CBS + CAL. "
            "Per methodology decision, orthogonal CAL-only definition is preserved."
        ),
    },
    "E3": {
        "name": "E3",
        "description": "Reliability-Weighted Consistency (RWC only)",
        "cbs_enabled": False,
        "cal_enabled": False,
        "rwc_enabled": True,
        "architecture_type": "ablation_single",
        "module": "src.training.train_loco_ccarc",
    },
    "E4": {
        "name": "E4",
        "description": "Full CCARC System (CBS + CAL + RWC)",
        "cbs_enabled": True,
        "cal_enabled": True,
        "rwc_enabled": True,
        "architecture_type": "full_proposed",
        "module": "src.training.train_loco_ccarc",
    },
}


@dataclass
class ExperimentStatus:
    """Detailed status report for a single variant and LOCO fold run."""

    variant: str
    fold_idx: int
    target_country: str
    status: str  # COMPLETED, IN_PROGRESS, PENDING, SMOKE_TEST_CHECKPOINT, INCOMPLETE
    action: str  # SKIP, RESUME, TRAIN, BLOCKED
    reason: str
    experiment_dir: str
    best_checkpoint: Optional[str] = None
    last_checkpoint: Optional[str] = None
    completed_epochs: int = 0
    target_epochs: int = 100
    cbs_enabled: bool = False
    cal_enabled: bool = False
    rwc_enabled: bool = False
    command: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AblationOrchestrator:
    """Orchestrates CCARC ablation experiments with strict checkpoint protection."""

    def __init__(
        self,
        views_dir: Union[str, Path] = DEFAULT_LOCAL_VIEWS_DIR,
        ccarc_dir: Union[str, Path] = DEFAULT_LOCAL_CCARC_DIR,
        baseline_dir: Union[str, Path] = DEFAULT_LOCAL_BASELINE_DIR,
        variants: Sequence[str] = ("E1", "E2", "E3", "E4"),
        folds: Sequence[int] = (0, 1, 2, 3, 4, 5),
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        workers: int = 2,
        patience: int = 20,
        optimizer: str = "SGD",
        lr0: float = 0.01,
        device: str = "0",
        allow_resume: bool = False,
        all_stages: bool = True,
        local_run_dir: Optional[Union[str, Path]] = None,
        colab_paths: bool = False,
    ):
        self.colab_paths = colab_paths
        if colab_paths:
            self.views_dir = COLAB_VIEWS_DIR
            self.ccarc_dir = COLAB_CCARC_DIR
            self.baseline_dir = COLAB_BASELINE_DIR
            self.local_run_dir = COLAB_LOCAL_RUN_DIR if local_run_dir is None else Path(local_run_dir)
        else:
            self.views_dir = Path(views_dir)
            self.ccarc_dir = Path(ccarc_dir)
            self.baseline_dir = Path(baseline_dir)
            self.local_run_dir = Path(local_run_dir) if local_run_dir else None

        self.variants = [v.upper() for v in variants]
        for v in self.variants:
            if v not in VALID_VARIANTS:
                raise ValueError(f"Invalid variant '{v}'. Must be one of {VALID_VARIANTS}.")

        self.folds = list(folds)
        for f in self.folds:
            if f not in range(len(LOCO_COUNTRIES)):
                raise ValueError(f"Invalid fold index '{f}'. Must be 0..{len(LOCO_COUNTRIES)-1}.")

        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.workers = workers
        self.patience = patience
        self.optimizer = optimizer
        self.lr0 = lr0
        self.device = str(device)
        self.allow_resume = allow_resume
        self.all_stages = all_stages

    def resolve_experiment_dir(self, variant: str, fold_idx: int) -> Path:
        """Resolve the target directory path for a variant and fold."""
        country = LOCO_COUNTRIES[fold_idx]
        if variant == "E0":
            # Baseline uses fold_{fold}_{country}
            return self.baseline_dir / f"fold_{fold_idx}_{country}"
        else:
            # CCARC variants use fold_{fold}_{country}_{variant}
            return self.ccarc_dir / f"fold_{fold_idx}_{country}_{variant}"

    def inspect_experiment_status(
        self, variant: str, fold_idx: int
    ) -> Tuple[str, str, str, Optional[Path], Optional[Path], int, int]:
        """Inspect the current disk state of an experiment run.

        Differentiates:
        - COMPLETED: Verified via train_stage_summary.json/fold_master_summary.json (including
          early-stopped runs) or complete results.csv (>= target_epochs) with best.pt.
        - IN_PROGRESS: Active or interrupted mid-run (last.pt present or results.csv < target_epochs
          without completion summary).
        - SMOKE_TEST_CHECKPOINT: Checkpoint originated from mini-debugging smoke batches.
        - PENDING: Directory missing or empty.
        - INCOMPLETE: Directory exists but required checkpoints missing.

        Returns:
            Tuple of (status, action, reason, best_ckpt, last_ckpt, completed_epochs, target_epochs).
        """
        exp_dir = self.resolve_experiment_dir(variant, fold_idx)
        best_ckpt: Optional[Path] = None
        last_ckpt: Optional[Path] = None

        if (exp_dir / "weights" / "best.pt").is_file():
            best_ckpt = exp_dir / "weights" / "best.pt"
        elif (exp_dir / "best.pt").is_file():
            best_ckpt = exp_dir / "best.pt"

        if (exp_dir / "weights" / "last.pt").is_file():
            last_ckpt = exp_dir / "weights" / "last.pt"
        elif (exp_dir / "last.pt").is_file():
            last_ckpt = exp_dir / "last.pt"

        if not exp_dir.is_dir():
            return (
                "PENDING",
                "TRAIN",
                "Directory does not exist. Experiment ready to train from scratch.",
                None,
                None,
                0,
                self.epochs,
            )

        # 1. Check for smoke test markers
        for smoke_file in ("smoke_master_summary.json",):
            if (exp_dir / smoke_file).is_file():
                return (
                    "SMOKE_TEST_CHECKPOINT",
                    "TRAIN",
                    f"Previous run was a smoke test ({smoke_file} present). Cleared for full training.",
                    best_ckpt,
                    last_ckpt,
                    0,
                    self.epochs,
                )

        # 2. Check completion summaries (written only when training exits cleanly, including early stopping)
        for summary_name in ("train_stage_summary.json", "fold_master_summary.json"):
            summary_path = exp_dir / summary_name
            if summary_path.is_file():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    is_smoke = (
                        data.get("smoke_mode") is True
                        or "SMOKE" in str(data.get("protocol", "")).upper()
                        or "SMOKE" in str(data.get("stage", "")).upper()
                        or data.get("smoke_batches_executed") is not None
                    )
                    if is_smoke:
                        return (
                            "SMOKE_TEST_CHECKPOINT",
                            "TRAIN",
                            f"Previous run was a smoke test summary in {summary_name}. Ready for full run.",
                            best_ckpt,
                            last_ckpt,
                            0,
                            self.epochs,
                        )
                    if best_ckpt and best_ckpt.is_file():
                        # Determine actual completed epochs if results.csv exists
                        actual_eps = self.epochs
                        results_csv = exp_dir / "results.csv"
                        if results_csv.is_file():
                            try:
                                with open(results_csv, "r", encoding="utf-8") as rf:
                                    lines = [l.strip() for l in rf if l.strip()]
                                actual_eps = max(0, len(lines) - 1) if len(lines) > 0 else self.epochs
                            except Exception:
                                pass
                        stopped_early = actual_eps < self.epochs
                        reason_msg = (
                            f"Experiment completed via early stopping at epoch {actual_eps}/{self.epochs} "
                            f"(verified by {summary_name}). Protected from overwrite."
                            if stopped_early
                            else f"Experiment completed all {actual_eps} epochs successfully (verified by {summary_name}). Protected from overwrite."
                        )
                        return (
                            "COMPLETED",
                            "SKIP",
                            reason_msg,
                            best_ckpt,
                            last_ckpt,
                            actual_eps,
                            self.epochs,
                        )
                except Exception:
                    pass

        # 3. Check results.csv row count vs target epochs (when summary json is absent)
        results_csv = exp_dir / "results.csv"
        completed_epochs = 0
        target_epochs = self.epochs

        args_yaml = exp_dir / "args.yaml"
        if args_yaml.is_file():
            try:
                with open(args_yaml, "r", encoding="utf-8") as f:
                    args_data = yaml.safe_load(f)
                    if isinstance(args_data, dict) and "epochs" in args_data:
                        target_epochs = int(args_data["epochs"])
            except Exception:
                pass

        if results_csv.is_file():
            try:
                with open(results_csv, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                completed_epochs = max(0, len(lines) - 1) if len(lines) > 0 else 0
            except Exception:
                completed_epochs = 0

            # Complete only if all epochs finished and best checkpoint exists
            if completed_epochs >= target_epochs and best_ckpt and best_ckpt.is_file():
                return (
                    "COMPLETED",
                    "SKIP",
                    f"Completed all {completed_epochs}/{target_epochs} epochs. Checkpoint protected from overwrite.",
                    best_ckpt,
                    last_ckpt,
                    completed_epochs,
                    target_epochs,
                )
            elif completed_epochs < target_epochs:
                # Interrupted run: no summary file and epochs < target
                if last_ckpt and last_ckpt.is_file():
                    if self.allow_resume:
                        return (
                            "IN_PROGRESS",
                            "RESUME",
                            f"Interrupted run ({completed_epochs}/{target_epochs} epochs; last.pt verified). Resume enabled.",
                            best_ckpt,
                            last_ckpt,
                            completed_epochs,
                            target_epochs,
                        )
                    else:
                        return (
                            "IN_PROGRESS",
                            "BLOCKED",
                            f"Interrupted or active run ({completed_epochs}/{target_epochs} epochs; last.pt present). Protected against overwrite.",
                            best_ckpt,
                            last_ckpt,
                            completed_epochs,
                            target_epochs,
                        )
                else:
                    return (
                        "INCOMPLETE",
                        "BLOCKED",
                        f"Interrupted run ({completed_epochs}/{target_epochs} epochs) but weights/last.pt is missing.",
                        best_ckpt,
                        None,
                        completed_epochs,
                        target_epochs,
                    )

        # 4. Check last.pt without results.csv
        if last_ckpt and last_ckpt.is_file():
            if self.allow_resume:
                return (
                    "IN_PROGRESS",
                    "RESUME",
                    "Found weights/last.pt without completion summary. Resume enabled.",
                    best_ckpt,
                    last_ckpt,
                    completed_epochs,
                    target_epochs,
                )
            else:
                return (
                    "IN_PROGRESS",
                    "BLOCKED",
                    "Found weights/last.pt without completion summary. Protected against destructive overwrite.",
                    best_ckpt,
                    last_ckpt,
                    completed_epochs,
                    target_epochs,
                )

        # 5. Check if best.pt exists without any completion summary or results.csv
        if best_ckpt and best_ckpt.is_file():
            return (
                "IN_PROGRESS",
                "BLOCKED",
                "Intermediate weights/best.pt present without completion summary. Protected against unconfirmed overwrite.",
                best_ckpt,
                last_ckpt,
                completed_epochs,
                target_epochs,
            )

        return (
            "PENDING",
            "TRAIN",
            "Directory empty or uninitialized. Ready to train from scratch.",
            None,
            None,
            0,
            target_epochs,
        )

    def build_command(
        self,
        variant: str,
        fold_idx: int,
        action: str,
    ) -> str:
        """Construct the exact CLI command for executing or resuming the experiment."""
        spec = VARIANT_SPECIFICATIONS[variant]
        module = spec["module"]

        output_dir_str = (
            f'"{self.baseline_dir.as_posix()}"'
            if variant == "E0"
            else f'"{self.ccarc_dir.as_posix()}"'
        )
        views_dir_str = f'"{self.views_dir.as_posix()}"'

        cmd_parts = [
            f"python -m {module}",
            f"--views-dir {views_dir_str}",
            f"--output-dir {output_dir_str}",
            f"--fold {fold_idx}",
        ]

        if variant != "E0":
            cmd_parts.append(f"--variant {variant}")

        cmd_parts.extend([
            f"--epochs {self.epochs}",
            f"--batch-size {self.batch_size}",
            f"--img-size {self.img_size}",
            f"--workers {self.workers}",
            f"--patience {self.patience}",
            f"--optimizer {self.optimizer}",
            f"--lr0 {self.lr0}",
            f"--device {self.device}",
        ])

        if self.all_stages:
            cmd_parts.append("--all-stages")

        if self.local_run_dir is not None:
            cmd_parts.append(f'--local-run-dir "{self.local_run_dir.as_posix()}"')

        # Only append --resume when explicitly resuming an in-progress run
        if action == "RESUME":
            cmd_parts.append("--resume")

        return " \\\n    ".join(cmd_parts)

    def plan(self) -> List[ExperimentStatus]:
        """Generate the full execution and safety plan across selected variants and folds."""
        plan_list: List[ExperimentStatus] = []
        for fold_idx in self.folds:
            target_country = LOCO_COUNTRIES[fold_idx]
            for variant in self.variants:
                spec = VARIANT_SPECIFICATIONS[variant]
                exp_dir = self.resolve_experiment_dir(variant, fold_idx)
                (
                    status,
                    action,
                    reason,
                    best_ckpt,
                    last_ckpt,
                    completed_epochs,
                    target_epochs,
                ) = self.inspect_experiment_status(variant, fold_idx)

                cmd = self.build_command(variant, fold_idx, action)

                status_rec = ExperimentStatus(
                    variant=variant,
                    fold_idx=fold_idx,
                    target_country=target_country,
                    status=status,
                    action=action,
                    reason=reason,
                    experiment_dir=exp_dir.as_posix(),
                    best_checkpoint=best_ckpt.as_posix() if best_ckpt else None,
                    last_checkpoint=last_ckpt.as_posix() if last_ckpt else None,
                    completed_epochs=completed_epochs,
                    target_epochs=target_epochs,
                    cbs_enabled=spec["cbs_enabled"],
                    cal_enabled=spec["cal_enabled"],
                    rwc_enabled=spec["rwc_enabled"],
                    command=cmd,
                )
                plan_list.append(status_rec)
        return plan_list

    def generate_markdown_report(self, plan_list: List[ExperimentStatus]) -> str:
        """Produce a clean GitHub Flavored Markdown summary report of the orchestration plan."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "# CCARC Ablation Experiment Orchestration Report",
            "",
            f"**Generated:** {timestamp}  ",
            f"**Execution Mode:** {'Google Colab Pro Paths' if self.colab_paths else 'Configured Workspace Paths'}  ",
            f"**Target Device:** `{self.device}` | **Epochs:** {self.epochs} | **Batch Size:** {self.batch_size} | **Workers:** {self.workers}  ",
            f"**Optimizer:** `{self.optimizer}` | **lr0:** {self.lr0} | **Patience:** {self.patience}  ",
            "",
            "## 1. Architectural Component Specifications & Audit",
            "",
            "The table below details component activation per ablation variant in `src/training/ccarc_config.py`:",
            "",
            "| Variant | Description | CBS | CAL | RWC | Codebase Status |",
            "|---|---|:---:|:---:|:---:|---|",
        ]

        for v in ("E0", "E1", "E2", "E3", "E4"):
            spec = VARIANT_SPECIFICATIONS[v]
            cbs_str = "✓" if spec["cbs_enabled"] else "—"
            cal_str = "✓" if spec["cal_enabled"] else "—"
            rwc_str = "✓" if spec["rwc_enabled"] else "—"
            note = "CAL-only (Isolated Ablation)" if v == "E2" else spec["architecture_type"]
            lines.append(
                f"| **{v}** | {spec['description']} | {cbs_str} | {cal_str} | {rwc_str} | {note} |"
            )

        lines.extend([
            "",
            "> [!NOTE]",
            "> **Ablation Methodology (Orthogonal Single-Component Design):**",
            "> In `src/training/ccarc_config.py`, E1 is CBS only, E2 is CAL only (`cbs=False, cal=True, rwc=False`),",
            "> E3 is RWC only, and E4 combines all three. This strictly isolates each component's individual effect.",
            "",
            "## 2. Planned Experiment Status & Checkpoint Safety",
            "",
            "| Fold | Target Country | Variant | Status | Action | Progress | Checkpoint Protection |",
            "|:---:|---|:---:|:---:|:---:|:---:|---|",
        ])

        for p in plan_list:
            status_badge = {
                "COMPLETED": "🟢 COMPLETED",
                "IN_PROGRESS": "🟡 IN_PROGRESS",
                "PENDING": "⚪ PENDING",
                "SMOKE_TEST_CHECKPOINT": "🟣 SMOKE_RUN",
                "INCOMPLETE": "🔴 INCOMPLETE",
            }.get(p.status, p.status)

            action_badge = {
                "SKIP": "🛡️ SKIP (Protected)",
                "RESUME": "🔄 RESUME (--resume)",
                "TRAIN": "🚀 TRAIN",
                "BLOCKED": "🔒 BLOCKED (Active/Interrupted)",
            }.get(p.action, p.action)

            prog_str = f"{p.completed_epochs}/{p.target_epochs} eps"
            lines.append(
                f"| Fold {p.fold_idx} | {p.target_country} | **{p.variant}** | {status_badge} | {action_badge} | {prog_str} | {p.reason} |"
            )

        lines.extend([
            "",
            "## 3. Generated Execution Commands",
            "",
            "Commands generated for pending or runnable experiments:",
            "",
        ])

        runnable = [p for p in plan_list if p.action in ("TRAIN", "RESUME")]
        if not runnable:
            lines.append("*(No experiments currently require execution. All runs are completed or safely protected.)*")
        else:
            for p in runnable:
                action_desc = "Train from scratch" if p.action == "TRAIN" else f"Resume from epoch {p.completed_epochs}"
                lines.extend([
                    f"### Fold {p.fold_idx} ({p.target_country}) — Variant {p.variant} ({action_desc})",
                    "```bash",
                    p.command,
                    "```",
                    "",
                ])

        return "\n".join(lines)

    def write_reports(
        self,
        output_dir: Union[str, Path],
        plan_list: List[ExperimentStatus],
    ) -> Tuple[Path, Path]:
        """Save JSON and Markdown status reports."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / "ablation_orchestration_status.json"
        md_file = out_path / "ablation_orchestration_summary.md"

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "colab_paths": self.colab_paths,
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "img_size": self.img_size,
            "workers": self.workers,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "views_dir": self.views_dir.as_posix(),
            "ccarc_dir": self.ccarc_dir.as_posix(),
            "baseline_dir": self.baseline_dir.as_posix(),
            "plan": [p.to_dict() for p in plan_list],
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        md_content = self.generate_markdown_report(plan_list)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_file, md_file

    def execute_plan(
        self,
        plan_list: List[ExperimentStatus],
        dry_run: bool = True,
        stop_on_error: bool = True,
    ) -> Dict[str, Any]:
        """Execute planned experiments sequentially or perform dry run."""
        results: Dict[str, Any] = {
            "dry_run": dry_run,
            "executed": [],
            "skipped": [],
            "failed": [],
        }

        for item in plan_list:
            key = f"fold_{item.fold_idx}_{item.target_country}_{item.variant}"
            if item.action in ("SKIP", "BLOCKED"):
                results["skipped"].append({
                    "experiment": key,
                    "reason": item.reason,
                    "status": item.status,
                })
                continue

            if dry_run:
                results["executed"].append({
                    "experiment": key,
                    "action": item.action,
                    "command": item.command,
                    "dry_run": True,
                })
                continue

            # Real execution
            print(f"\n[ORCHESTRATOR] Starting {key} ({item.action})...")
            clean_cmd = " ".join([part.strip().rstrip("\\") for part in item.command.splitlines()])
            try:
                ret = subprocess.run(clean_cmd, shell=True, check=True)
                results["executed"].append({
                    "experiment": key,
                    "action": item.action,
                    "exit_code": ret.returncode,
                })
            except subprocess.CalledProcessError as err:
                results["failed"].append({
                    "experiment": key,
                    "action": item.action,
                    "exit_code": err.returncode,
                    "error": str(err),
                })
                if stop_on_error:
                    break

        return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCARC Ablation Experiment Orchestrator and Checkpoint Safety Controller"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["E1", "E2", "E3"],
        choices=list(VALID_VARIANTS),
        help="Ablation variants to orchestrate (default: E1 E2 E3).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[4],
        choices=[0, 1, 2, 3, 4, 5],
        help="LOCO fold indices (default: 4 for Norway).",
    )
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=DEFAULT_LOCAL_VIEWS_DIR,
        help="Directory containing runtime LOCO views.",
    )
    parser.add_argument(
        "--ccarc-dir",
        type=Path,
        default=DEFAULT_LOCAL_CCARC_DIR,
        help="Directory for CCARC ablation experiment outputs.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_LOCAL_BASELINE_DIR,
        help="Directory containing baseline E0 experiment outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("experiments/orchestration_reports"),
        help="Directory to write JSON and Markdown status reports.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Target training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image resolution (default: 640).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes (default: 2).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Optimizer (default: 'SGD').",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Execution device ('0', 'cuda', 'cpu', default: '0').",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Format commands and paths using standard Google Colab Pro layout.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow resuming in-progress experiments that have last.pt.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the planned experiments sequentially (default is safe dry-run).",
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        default=True,
        help="Run Stage 1 training, Stage 2 source validation, and Stage 3 held-out test (default: True).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    orchestrator = AblationOrchestrator(
        views_dir=args.views_dir,
        ccarc_dir=args.ccarc_dir,
        baseline_dir=args.baseline_dir,
        variants=args.variants,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        device=args.device,
        allow_resume=args.resume,
        all_stages=args.all_stages,
        colab_paths=args.colab,
    )

    plan = orchestrator.plan()
    report_json, report_md = orchestrator.write_reports(args.report_dir, plan)

    print("\n" + "=" * 80)
    print("CCARC Ablation Experiment Orchestration Plan")
    print("=" * 80)
    print(orchestrator.generate_markdown_report(plan))
    print("=" * 80)
    print(f"Status reports saved to:\n  JSON: {report_json}\n  Markdown: {report_md}")

    if args.execute:
        print("\nExecuting planned experiments sequentially...")
        res = orchestrator.execute_plan(plan, dry_run=False)
        print(f"Execution finished: {len(res['executed'])} succeeded, {len(res['failed'])} failed.")
        if res["failed"]:
            return 1
    else:
        print("\nDry-run completed successfully. (Pass --execute to execute runnable experiments)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
