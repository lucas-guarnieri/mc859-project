"""
Generate two report-ready plots from experiment results.
  Plot 1: Precision@10 by model and period (all vs rated panels)
  Plot 2: Precision vs Diversity scatter with model categories
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

METRICS_DIR = "results/metrics"
OUTPUT_DIR = "results/plots"

PERIODS = [
    ("2015_2016", "2015→2016"),
    ("2016_2017", "2016→2017"),
    ("2017_2018", "2017→2018"),
]

MODEL_DISPLAY = {
    "popularity":                    "Popularidade",
    "cooccurrence":                  "Co-ocorrência",
    "item_cf":                       "Item-CF",
    "pagerank":                      "PageRank",
    "multi_hop_3":                   "Multi-hop",
    "multi_hop_penalized_sqrt_3":    "MH-sqrt",
    "multi_hop_penalized_log_3":     "MH-log",
    "multi_hop_penalized_quadratic_3": "MH-quad",
}

# Category: (color, group label)
MODEL_CATEGORY = {
    "popularity":                    ("baselines",  "#9e9e9e"),
    "cooccurrence":                  ("baselines",  "#bdbdbd"),
    "item_cf":                       ("baselines",  "#757575"),
    "pagerank":                      ("referência", "#1565c0"),
    "multi_hop_3":                   ("propostos",  "#e65100"),
    "multi_hop_penalized_sqrt_3":    ("propostos",  "#ef6c00"),
    "multi_hop_penalized_log_3":     ("propostos",  "#f57c00"),
    "multi_hop_penalized_quadratic_3": ("propostos","#fb8c00"),
}

MODEL_ORDER = [
    "popularity",
    "cooccurrence",
    "item_cf",
    "pagerank",
    "multi_hop_3",
    "multi_hop_penalized_quadratic_3",
    "multi_hop_penalized_sqrt_3",
    "multi_hop_penalized_log_3",
]

PERIOD_COLORS = ["#1976d2", "#388e3c", "#7b1fa2"]


def load_data():
    data = {}
    for period_key, _ in PERIODS:
        data[period_key] = {}
        for mode in ("all", "rated"):
            path = os.path.join(METRICS_DIR, f"results_{period_key}_{mode}.json")
            with open(path) as f:
                data[period_key][mode] = json.load(f)
    return data


def plot_precision_by_period(data, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.suptitle("Precision@10 por Modelo e Período", fontsize=14, fontweight="bold", y=1.01)

    modes = [("all", "Modo: todas as interações"), ("rated", "Modo: avaliações ≥ 4.0")]
    n_models = len(MODEL_ORDER)
    n_periods = len(PERIODS)
    bar_w = 0.22
    group_gap = 0.08
    group_w = n_periods * bar_w + group_gap
    x = np.arange(n_models) * group_w

    for ax, (mode, mode_title) in zip(axes, modes):
        for pi, (period_key, period_label) in enumerate(PERIODS):
            precisions = [
                data[period_key][mode].get(m, {}).get("precision", 0)
                for m in MODEL_ORDER
            ]
            offset = (pi - (n_periods - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset, precisions,
                width=bar_w * 0.92,
                color=PERIOD_COLORS[pi],
                label=period_label,
                zorder=3,
            )

        ax.set_title(mode_title, fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY[m] for m in MODEL_ORDER],
            rotation=30, ha="right", fontsize=9
        )
        ax.set_ylabel("Precision@10", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Shade model category regions
        category_bands = [
            (0, 2, "#f5f5f5"),   # baselines
            (3, 3, "#e3f2fd"),   # referência
            (4, 7, "#fff3e0"),   # propostos
        ]
        for start, end, color in category_bands:
            ax.axvspan(
                x[start] - group_w / 2,
                x[end] + group_w / 2,
                alpha=0.35, color=color, zorder=0
            )

        if ax == axes[1]:
            ax.legend(title="Período de treino→teste", fontsize=8,
                      title_fontsize=8, loc="upper right")

    # Category legend patches
    cat_patches = [
        mpatches.Patch(color="#e0e0e0", label="Baselines clássicos"),
        mpatches.Patch(color="#bbdefb", label="Referência de grafo (PageRank)"),
        mpatches.Patch(color="#ffe0b2", label="Modelos propostos (multi-hop)"),
    ]
    fig.legend(handles=cat_patches, loc="lower center", ncol=3,
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.07))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_precision_diversity_scatter(data, output_path):
    # Average precision and diversity across all 6 scenarios per model
    avg = {m: {"precision": [], "diversity": []} for m in MODEL_ORDER}
    for period_key, _ in PERIODS:
        for mode in ("all", "rated"):
            for m in MODEL_ORDER:
                entry = data[period_key][mode].get(m, {})
                avg[m]["precision"].append(entry.get("precision", 0))
                avg[m]["diversity"].append(entry.get("diversity_score", 0))

    means = {
        m: {
            "precision": np.mean(avg[m]["precision"]),
            "diversity": np.mean(avg[m]["diversity"]),
        }
        for m in MODEL_ORDER
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    cat_colors = {
        "baselines":  "#9e9e9e",
        "referência": "#1565c0",
        "propostos":  "#e65100",
    }
    cat_markers = {
        "baselines":  "s",
        "referência": "D",
        "propostos":  "o",
    }

    for m in MODEL_ORDER:
        cat, _ = MODEL_CATEGORY[m]
        prec = means[m]["precision"]
        div  = means[m]["diversity"]
        ax.scatter(
            prec, div,
            color=cat_colors[cat],
            marker=cat_markers[cat],
            s=90, zorder=4, edgecolors="white", linewidths=0.6,
        )
        # Label offset to avoid overlap
        dx, dy = 0.00005, 0.004
        if m == "multi_hop_penalized_log_3":
            dx = -0.00015; dy = -0.012
        elif m == "multi_hop_penalized_sqrt_3":
            dx = 0.00005; dy = -0.012
        elif m == "multi_hop_penalized_quadratic_3":
            dx = 0.00005; dy = 0.006
        elif m == "pagerank":
            dx = 0.00005; dy = 0.006
        elif m == "item_cf":
            dx = -0.00008; dy = 0.006
        ax.annotate(
            MODEL_DISPLAY[m],
            (prec, div),
            xytext=(prec + dx, div + dy),
            fontsize=8.5,
            color="#333333",
        )

    ax.set_xlabel("Precision@10 (média sobre 6 cenários)", fontsize=11)
    ax.set_ylabel("Diversidade (média sobre 6 cenários)", fontsize=11)
    ax.set_title("Trade-off entre Precisão e Diversidade", fontsize=13, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=cat_colors["baselines"],
               markersize=9, label="Baselines clássicos"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=cat_colors["referência"],
               markersize=9, label="Referência de grafo (PageRank)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_colors["propostos"],
               markersize=9, label="Modelos propostos (multi-hop)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    # Annotate quadrants
    xmid = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.55
    ymid = 0.80
    ax.axvline(xmid, color="#cccccc", linestyle=":", linewidth=0.8, zorder=1)
    ax.axhline(ymid, color="#cccccc", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(xmid + 0.00002, ymid + 0.005, "alta precisão\nbaixa diversidade",
            fontsize=7, color="#888888", ha="left")
    ax.text(ax.get_xlim()[0] + 0.00001, ymid + 0.005, "baixa precisão\nalta diversidade",
            fontsize=7, color="#888888", ha="left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_data()

    print("Gerando gráficos para o relatório...")
    plot_precision_by_period(
        data,
        os.path.join(OUTPUT_DIR, "fig1_precision_by_period.png")
    )
    plot_precision_diversity_scatter(
        data,
        os.path.join(OUTPUT_DIR, "fig2_precision_diversity_tradeoff.png")
    )
    print("Concluído.")


if __name__ == "__main__":
    main()
