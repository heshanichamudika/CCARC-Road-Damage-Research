"""Report Generation Module for CCARC LOCO Benchmark Evaluations.

Generates:
1. Machine-readable JSON summary (`loco_benchmark_results.json`)
2. Machine-readable CSV summary (`loco_benchmark_summary.csv`)
3. Per-class metrics breakdown CSV (`loco_benchmark_per_class.csv`)
4. Markdown comparison table for research reports (`loco_benchmark_summary.md`)
5. LaTeX publication table for academic manuscripts (`loco_benchmark_summary.tex`)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.training.ccarc_config import VALID_VARIANTS

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "E0": "Baseline (YOLOv8s)",
    "E1": "+ CBS (Country-Balanced Sampler)",
    "E2": "+ CAL (Class-Aware Loss)",
    "E3": "+ RWC (Reliability Consistency)",
    "E4": "Full CCARC (CBS + CAL + RWC)",
}


class EvaluationReportGenerator:
    """Formats, aggregates, and exports CCARC benchmark evaluation results."""

    def __init__(
        self,
        results_data: Dict[str, Any],
        target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
        variants: Sequence[str] = VALID_VARIANTS,
        countries: Sequence[str] = LOCO_COUNTRIES,
    ) -> None:
        """Initialize generator with structured evaluation results."""
        self.raw_data = results_data
        self.experiments: Dict[str, Dict[str, Any]] = results_data.get("experiments", {})
        self.target_classes = list(target_classes)
        self.variants = list(variants)
        self.countries = list(countries)

    def export_all(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Export JSON, CSVs, Markdown, and LaTeX summaries to target directory."""
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "loco_benchmark_results.json"
        summary_csv_path = out_dir / "loco_benchmark_summary.csv"
        per_class_csv_path = out_dir / "loco_benchmark_per_class.csv"
        md_path = out_dir / "loco_benchmark_summary.md"
        tex_path = out_dir / "loco_benchmark_summary.tex"

        self.export_json(json_path)
        self.export_csv(summary_csv_path, per_class_csv_path)
        self.export_markdown(md_path)
        self.export_latex(tex_path)

        return {
            "json": json_path,
            "summary_csv": summary_csv_path,
            "per_class_csv": per_class_csv_path,
            "markdown": md_path,
            "latex": tex_path,
        }

    def export_json(self, path: Union[str, Path]) -> None:
        """Export full machine-readable JSON benchmark results."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.raw_data, f, indent=2)

    def export_csv(
        self,
        summary_path: Union[str, Path],
        per_class_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Export tabular evaluation metrics to CSV files."""
        summary_rows = []
        per_class_rows = []

        for exp_key, exp in self.experiments.items():
            var = exp.get("variant", "")
            fold = exp.get("fold_idx", -1)
            country = exp.get("target_country", "")
            status = exp.get("status", "UNKNOWN")
            ckpt = exp.get("checkpoint_path", "")
            metrics = exp.get("metrics") or {}

            # Summary row
            row = {
                "variant": var,
                "fold": fold,
                "target_country": country,
                "status": status,
                "checkpoint": ckpt,
                "precision": metrics.get("precision", np.nan),
                "recall": metrics.get("recall", np.nan),
                "f1": metrics.get("f1", np.nan),
                "map50": metrics.get("map50", np.nan),
                "map50_95": metrics.get("map50_95", np.nan),
                "fitness": metrics.get("fitness", np.nan),
            }

            per_class_data = metrics.get("per_class", {})
            for cls_name in self.target_classes:
                c_data = per_class_data.get(cls_name, {})
                row[f"{cls_name}_ap50"] = c_data.get("ap50", np.nan)
                row[f"{cls_name}_map50_95"] = c_data.get("map50_95", np.nan)

                # Per-class row
                per_class_rows.append({
                    "variant": var,
                    "fold": fold,
                    "target_country": country,
                    "status": status,
                    "class": cls_name,
                    "precision": c_data.get("precision", np.nan),
                    "recall": c_data.get("recall", np.nan),
                    "f1": c_data.get("f1", np.nan),
                    "ap50": c_data.get("ap50", np.nan),
                    "map50_95": c_data.get("map50_95", np.nan),
                })

            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(summary_path, index=False)

        df_class = None
        if per_class_path is not None:
            df_class = pd.DataFrame(per_class_rows)
            Path(per_class_path).parent.mkdir(parents=True, exist_ok=True)
            df_class.to_csv(per_class_path, index=False)

        return df_summary, df_class

    def generate_markdown_table(self, path: Optional[Union[str, Path]] = None) -> str:
        """Generate structured markdown comparison tables for research reports."""
        lines: List[str] = []
        lines.append("# CCARC Cross-Country Road Damage Detection Benchmark")
        lines.append(f"\n*Generated at: {datetime.now(timezone.utc).isoformat()}*\n")

        # 1. Primary mAP@0.5 Comparison Table
        lines.append("## 1. Held-Out Target Generalization (mAP@0.5)\n")
        header = ["Fold", "Held-Out Country"] + [f"{v} ({VARIANT_DESCRIPTIONS.get(v, v)})" for v in self.variants]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        fold_scores_map50: Dict[str, List[float]] = {v: [] for v in self.variants}
        fold_scores_map50_95: Dict[str, List[float]] = {v: [] for v in self.variants}

        for fold_idx, country in enumerate(self.countries):
            row = [str(fold_idx), country]
            for v in self.variants:
                exp_key = f"{v}_fold_{fold_idx}_{country}"
                exp = self.experiments.get(exp_key, {})
                metrics = exp.get("metrics")
                if metrics and metrics.get("map50") is not None:
                    val = float(metrics["map50"])
                    fold_scores_map50[v].append(val)
                    row.append(f"{val:.4f}")
                else:
                    status = exp.get("status", "PENDING")
                    row.append(f"*{status}*")
            lines.append("| " + " | ".join(row) + " |")

        # Mean row for mAP@0.5
        mean_row = ["**Mean**", "**Macro Average**"]
        for v in self.variants:
            scores = fold_scores_map50[v]
            if scores:
                mean_row.append(f"**{np.mean(scores):.4f}** ({len(scores)}/{len(self.countries)})")
            else:
                mean_row.append("*N/A*")
        lines.append("| " + " | ".join(mean_row) + " |")

        # 2. Strict mAP@0.5:0.95 Comparison Table
        lines.append("\n## 2. Strict Detection Accuracy (mAP@0.5:0.95)\n")
        header_95 = ["Fold", "Held-Out Country"] + [f"{v}" for v in self.variants]
        lines.append("| " + " | ".join(header_95) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_95)) + " |")

        for fold_idx, country in enumerate(self.countries):
            row = [str(fold_idx), country]
            for v in self.variants:
                exp_key = f"{v}_fold_{fold_idx}_{country}"
                exp = self.experiments.get(exp_key, {})
                metrics = exp.get("metrics")
                if metrics and metrics.get("map50_95") is not None:
                    val = float(metrics["map50_95"])
                    fold_scores_map50_95[v].append(val)
                    row.append(f"{val:.4f}")
                else:
                    status = exp.get("status", "PENDING")
                    row.append(f"*{status}*")
            lines.append("| " + " | ".join(row) + " |")

        mean_row_95 = ["**Mean**", "**Macro Average**"]
        for v in self.variants:
            scores = fold_scores_map50_95[v]
            if scores:
                mean_row_95.append(f"**{np.mean(scores):.4f}** ({len(scores)}/{len(self.countries)})")
            else:
                mean_row_95.append("*N/A*")
        lines.append("| " + " | ".join(mean_row_95) + " |")

        # 3. Overall F1-Score Table
        lines.append("\n## 3. Harmonic Balance (F1-Score)\n")
        header_f1 = ["Fold", "Held-Out Country"] + [f"{v}" for v in self.variants]
        lines.append("| " + " | ".join(header_f1) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_f1)) + " |")

        fold_scores_f1: Dict[str, List[float]] = {v: [] for v in self.variants}
        for fold_idx, country in enumerate(self.countries):
            row = [str(fold_idx), country]
            for v in self.variants:
                exp_key = f"{v}_fold_{fold_idx}_{country}"
                exp = self.experiments.get(exp_key, {})
                metrics = exp.get("metrics")
                if metrics and metrics.get("f1") is not None:
                    val = float(metrics["f1"])
                    fold_scores_f1[v].append(val)
                    row.append(f"{val:.4f}")
                else:
                    row.append(f"*{exp.get('status', 'PENDING')}*")
            lines.append("| " + " | ".join(row) + " |")

        mean_row_f1 = ["**Mean**", "**Macro Average**"]
        for v in self.variants:
            scores = fold_scores_f1[v]
            if scores:
                mean_row_f1.append(f"**{np.mean(scores):.4f}** ({len(scores)}/{len(self.countries)})")
            else:
                mean_row_f1.append("*N/A*")
        lines.append("| " + " | ".join(mean_row_f1) + " |")

        # 4. Status Tracking Table
        lines.append("\n## 4. Experiment Status Matrix\n")
        lines.append("| Experiment Variant | Completed | In Progress | Pending | Total |")
        lines.append("| --- | --- | --- | --- | --- |")
        for v in self.variants:
            comp = sum(1 for f in range(len(self.countries)) if self.experiments.get(f"{v}_fold_{f}_{self.countries[f]}", {}).get("status") == "COMPLETED")
            in_prog = sum(1 for f in range(len(self.countries)) if self.experiments.get(f"{v}_fold_{f}_{self.countries[f]}", {}).get("status") == "IN_PROGRESS")
            pending = len(self.countries) - comp - in_prog
            lines.append(f"| **{v}** ({VARIANT_DESCRIPTIONS.get(v, '')}) | {comp} | {in_prog} | {pending} | {len(self.countries)} |")

        md_content = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).write_text(md_content, encoding="utf-8")

        return md_content

    def export_markdown(self, path: Union[str, Path]) -> None:
        """Write markdown report to path."""
        self.generate_markdown_table(path)

    def generate_latex_table(self, path: Optional[Union[str, Path]] = None) -> str:
        """Generate publication-ready LaTeX table using booktabs format."""
        lines: List[str] = []
        lines.append("% CCARC Leave-One-Country-Out Benchmark Comparison")
        lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{multirow}")
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\caption{Leave-One-Country-Out (LOCO) Cross-Country Road Damage Detection Benchmark on RDD2022. Values report held-out target domain test evaluation.}")
        lines.append("\\label{tab:ccarc_loco_benchmark}")

        col_spec = "ll" + "cc" * len(self.variants)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # First header row
        h1 = ["\\multirow{2}{*}{Fold}", "\\multirow{2}{*}{Held-Out Target}"]
        for v in self.variants:
            h1.append(f"\\multicolumn{{2}}{{c}}{{{v}}}")
        lines.append(" & ".join(h1) + " \\\\")

        # Second header row
        cmidrules = []
        col_idx = 3
        for _ in self.variants:
            cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
            col_idx += 2
        lines.append(" ".join(cmidrules))

        h2 = ["", ""]
        for _ in self.variants:
            h2.extend(["mAP@.5", "mAP@.5:.95"])
        lines.append(" & ".join(h2) + " \\\\")
        lines.append("\\midrule")

        scores_50: Dict[str, List[float]] = {v: [] for v in self.variants}
        scores_95: Dict[str, List[float]] = {v: [] for v in self.variants}

        for fold_idx, country in enumerate(self.countries):
            c_label = country.replace("_", " ")
            row = [str(fold_idx), c_label]
            for v in self.variants:
                exp_key = f"{v}_fold_{fold_idx}_{country}"
                exp = self.experiments.get(exp_key, {})
                metrics = exp.get("metrics")
                if metrics and metrics.get("map50") is not None and metrics.get("map50_95") is not None:
                    m50 = float(metrics["map50"])
                    m95 = float(metrics["map50_95"])
                    scores_50[v].append(m50)
                    scores_95[v].append(m95)
                    row.extend([f"{m50:.4f}", f"{m95:.4f}"])
                else:
                    row.extend(["--", "--"])
            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\midrule")

        # Mean macro average row
        mean_row = ["\\textbf{Mean}", "\\textbf{Macro Avg}"]
        for v in self.variants:
            s50 = scores_50[v]
            s95 = scores_95[v]
            m50_str = f"\\textbf{{{np.mean(s50):.4f}}}" if s50 else "--"
            m95_str = f"\\textbf{{{np.mean(s95):.4f}}}" if s95 else "--"
            mean_row.extend([m50_str, m95_str])
        lines.append(" & ".join(mean_row) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")

        tex_content = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).write_text(tex_content, encoding="utf-8")

        return tex_content

    def export_latex(self, path: Union[str, Path]) -> None:
        """Write LaTeX table to path."""
        self.generate_latex_table(path)
