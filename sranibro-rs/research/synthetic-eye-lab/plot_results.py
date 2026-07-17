"""Generate basic deterministic reports from a synthetic-eye-lab output directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_FIELDS = ("presence", "open_l", "open_r", "squeeze_l", "squeeze_r")


def number(value: str) -> float:
    if value == "+Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    return float(value)


def load_rows(run: Path) -> list[dict[str, object]]:
    with (run / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field in OUTPUT_FIELDS + (
                "factor_x",
                "factor_y",
                "mean_l",
                "stddev_l",
                "edge_energy_l",
                "visible_area_l",
                "match_sclera_level",
                "match_target_mean",
                "match_achieved_mean",
                "match_abs_error",
            ):
                value = raw.get(field, "")
                row[field] = number(value) if value else float("nan")
            row["usable"] = raw["usable_for_interpretation"].lower() == "true"
            row["match_at_bound"] = raw.get("match_at_bound", "").lower() == "true"
            rows.append(row)
        return rows


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def plot_1d(run: Path, grouped: dict[str, list[dict[str, object]]], summaries: dict[str, dict]) -> None:
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    for experiment, rows in grouped.items():
        if not rows[0]["factor_x_name"] or rows[0]["factor_y_name"]:
            continue
        ordered = sorted(rows, key=lambda row: float(row["factor_x"]))
        x = np.asarray([float(row["factor_x"]) for row in ordered])
        finite = np.asarray([all(np.isfinite(float(row[key])) for key in OUTPUT_FIELDS) for row in ordered])
        if finite.sum() < 2:
            continue
        x = x[finite]
        data = {key: np.asarray([float(row[key]) for row in ordered])[finite] for key in OUTPUT_FIELDS}
        usable = np.asarray([bool(row["usable"]) for row in ordered])[finite]
        excluded = ~usable

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        axes[0, 0].plot(x, data["presence"], marker="o", markersize=3)
        axes[0, 0].axhline(0.05, color="tab:red", linestyle="--", linewidth=1, label="production gate")
        axes[0, 0].set_ylabel("raw presence")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].plot(x, data["open_l"], label="left")
        axes[0, 1].plot(x, data["open_r"], label="right")
        axes[0, 1].set_ylabel("raw openness")
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].plot(x, data["squeeze_l"], label="left")
        axes[1, 0].plot(x, data["squeeze_r"], label="right")
        axes[1, 0].set_ylabel("raw squeeze")
        axes[1, 0].legend(fontsize=8)
        if excluded.any():
            axes[0, 0].scatter(
                x[excluded], data["presence"][excluded], marker="x", color="tab:red",
                zorder=5, label="excluded from summary",
            )
            axes[0, 0].legend(fontsize=8)
            for field, axis in (
                ("open_l", axes[0, 1]),
                ("open_r", axes[0, 1]),
                ("squeeze_l", axes[1, 0]),
                ("squeeze_r", axes[1, 0]),
            ):
                axis.scatter(
                    x[excluded], data[field][excluded], marker="x", color="tab:red", zorder=5
                )
        if len(x) >= 3 and np.all(np.diff(x) > 0):
            axes[1, 1].plot(x, np.gradient(data["open_l"], x), label="d left / dx")
            axes[1, 1].plot(x, np.gradient(data["open_r"], x), label="d right / dx")
            axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_ylabel("finite-difference sensitivity")
        for axis in axes.flat:
            axis.set_xlabel(str(rows[0]["factor_x_name"]))
            axis.grid(alpha=0.25)
        withheld = summaries.get(experiment, {}).get("interpretation_withheld", True)
        fig.suptitle(f"{experiment} - raw synthetic response" + (" (interpretation withheld)" if withheld else ""))
        fig.savefig(plots / f"{safe_name(experiment)}.png", dpi=150)
        plt.close(fig)


def plot_stretch(run: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    xs = sorted({float(row["factor_x"]) for row in rows})
    ys = sorted({float(row["factor_y"]) for row in rows})
    index = {(float(row["factor_x"]), float(row["factor_y"])): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, field in zip(axes, ("presence", "open_l", "open_r")):
        grid = np.asarray([[float(index[(x, y)][field]) for x in xs] for y in ys])
        image = axis.imshow(grid, origin="lower", aspect="auto", extent=(xs[0], xs[-1], ys[0], ys[-1]))
        axis.set_title(field)
        axis.set_xlabel("scale_x")
        axis.set_ylabel("scale_y")
        excluded = [row for row in rows if not row["usable"]]
        if excluded:
            axis.scatter(
                [float(row["factor_x"]) for row in excluded],
                [float(row["factor_y"]) for row in excluded],
                marker="x",
                color="red",
                linewidths=2,
                label="excluded: frame contact / saturation",
            )
            axis.legend(fontsize=7, loc="lower left")
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle("stretch_grid - raw response surface")
    fig.savefig(run / "plots" / "stretch_grid_heatmaps.png", dpi=150)
    plt.close(fig)


def plot_correlations(run: Path, grouped: dict[str, list[dict[str, object]]]) -> None:
    # Never pool heterogeneous interventions into one correlation matrix. Each plot is
    # descriptive only within one preregistered suite.
    for experiment, rows in grouped.items():
        usable = [
            row
            for row in rows
            if row["usable"] and all(np.isfinite(float(row[key])) for key in OUTPUT_FIELDS)
        ]
        if len(usable) < 5:
            continue
        values = np.asarray([[float(row[key]) for key in OUTPUT_FIELDS] for row in usable]).T
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.corrcoef(values)
        fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(OUTPUT_FIELDS)), OUTPUT_FIELDS, rotation=35, ha="right")
        axis.set_yticks(range(len(OUTPUT_FIELDS)), OUTPUT_FIELDS)
        axis.set_title(f"Output correlation - {experiment} usable cases")
        fig.colorbar(image, ax=axis, shrink=0.8)
        fig.savefig(
            run / "plots" / f"output_correlation_{safe_name(experiment)}.png", dpi=150
        )
        plt.close(fig)


def plot_luminance_match(run: Path, grouped: dict[str, list[dict[str, object]]]) -> None:
    names = (
        "aperture_geometry",
        "aperture_constant_mean",
        "fixed_geometry_same_sclera_control",
        "fixed_geometry_original_mean_control",
    )
    if not all(grouped.get(name) for name in names):
        return
    b_rows = grouped["aperture_constant_mean"]
    low = min(float(row["factor_x"]) for row in b_rows)
    high = max(float(row["factor_x"]) for row in b_rows)
    labels = {
        "aperture_geometry": "A original aperture",
        "aperture_constant_mean": "B constant mean",
        "fixed_geometry_same_sclera_control": "C fixed geometry / B sclera",
        "fixed_geometry_original_mean_control": "D fixed geometry / A mean",
    }
    colors = dict(zip(names, ("black", "tab:blue", "tab:orange", "tab:green")))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name in names:
        rows = sorted(grouped[name], key=lambda row: float(row["factor_x"]))
        rows = [row for row in rows if low - 1e-6 <= float(row["factor_x"]) <= high + 1e-6]
        x = np.asarray([float(row["factor_x"]) for row in rows])
        usable = np.asarray([bool(row["usable"]) for row in rows])
        open_average = np.asarray(
            [(float(row["open_l"]) + float(row["open_r"])) * 0.5 for row in rows]
        )
        axes[0, 0].plot(x, open_average, marker="o", markersize=3, color=colors[name], label=labels[name])
        axes[0, 1].plot(
            x, [float(row["mean_l"]) for row in rows], marker="o", markersize=3,
            color=colors[name], label=labels[name],
        )
        axes[1, 1].plot(
            x, [float(row["edge_energy_l"]) for row in rows], marker="o", markersize=3,
            color=colors[name], label=labels[name],
        )
        if name == "aperture_geometry":
            axes[1, 0].plot(x, np.full_like(x, 0.78), color=colors[name], label="A default sclera")
        else:
            sclera = np.asarray([float(row["match_sclera_level"]) for row in rows])
            axes[1, 0].plot(x, sclera, marker="o", markersize=3, color=colors[name], label=labels[name])
            pinned = np.asarray([bool(row["match_at_bound"]) for row in rows])
            if pinned.any():
                axes[1, 0].scatter(x[pinned], sclera[pinned], marker="x", color="red", zorder=6)
        excluded = ~usable
        if excluded.any():
            axes[0, 0].scatter(x[excluded], open_average[excluded], marker="x", color="red", zorder=6)

    axes[0, 0].set_ylabel("mean raw openness L/R")
    axes[0, 1].set_ylabel("whole-image mean")
    axes[1, 0].set_ylabel("sclera level (red x = bound-pinned)")
    axes[1, 1].set_ylabel("edge energy (unmatched confound)")
    for axis in axes.flat:
        axis.set_xlabel("source aperture")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("Phase 1.1 luminance match - signed A/B/C/D comparison")
    fig.savefig(run / "plots" / "luminance_match_comparison.png", dpi=160)
    plt.close(fig)


def contact_sheet(run: Path, rows: list[dict[str, object]]) -> None:
    selected: list[dict[str, object]] = []
    selected.extend(row for row in rows if row["experiment"] == "anchor_family")
    usable = [row for row in rows if row["usable"]]
    for experiment in (
        "aperture_geometry",
        "global_brightness_offset",
        "rotation",
        "stretch_grid",
        "aperture_constant_mean",
        "fixed_geometry_same_sclera_control",
        "fixed_geometry_original_mean_control",
    ):
        group = [row for row in usable if row["experiment"] == experiment]
        if group:
            selected.extend((min(group, key=lambda row: float(row["open_l"])), max(group, key=lambda row: float(row["open_l"]))))
    unique = {str(row["case_id"]): row for row in selected}
    selected = list(unique.values())[:12]
    if not selected:
        return
    columns = 3
    rows_n = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows_n, columns, figsize=(9, 3 * rows_n), squeeze=False, constrained_layout=True)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, selected):
        case = str(row["case_id"])
        left = mpimg.imread(run / "inputs" / f"{case}_left.png")
        right = mpimg.imread(run / "inputs" / f"{case}_right.png")
        axis.imshow(np.concatenate((left, right), axis=1), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{row['experiment']}\n{row['case_name']}", fontsize=8)
        axis.axis("off")
    out = run / "contact-sheets"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "selected_cases.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="synthetic-eye-lab output directory")
    args = parser.parse_args()
    run = args.run.resolve()
    rows = load_rows(run)
    with (run / "summary.json").open(encoding="utf-8") as handle:
        summaries = {item["experiment"]: item for item in json.load(handle)}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment"])].append(row)
    plot_1d(run, grouped, summaries)
    plot_stretch(run, grouped.get("stretch_grid", []))
    plot_correlations(run, grouped)
    plot_luminance_match(run, grouped)
    contact_sheet(run, rows)
    print(f"plots: {run / 'plots'}")
    print(f"contact sheets: {run / 'contact-sheets'}")


if __name__ == "__main__":
    main()
