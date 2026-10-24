"""Draws the README charts from the files in results/ (copied from the Kaggle notebook outputs)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from finsent import LABELS, check, rules_label  # noqa: E402

RES, DOCS = ROOT / "results", ROOT / "docs"
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a1"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated together
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
PCT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")


def load(path):
    return json.loads((RES / path).read_text(encoding="utf-8"))


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side == ("bottom" if axis == "y" else "left"))
    ax.tick_params(length=0)


def footer(fig, text):
    fig.text(0.01, 0.015, text, fontsize=8, color=INK2, ha="left", va="bottom")


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)


def line_panel(ax, xs, ys, heading, fmt, ylim, color=BLUE, start_right=False):
    ax.plot(xs, ys, color=color, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
    ax.scatter([xs[-1]], [ys[-1]], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(fmt(ys[-1]), (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color=INK, fontweight="bold")
    ax.annotate(fmt(ys[0]), (xs[0], ys[0]), xytext=(7, 4) if start_right else (-7, 0), textcoords="offset points",
                ha="left" if start_right else "right", va="center", fontsize=9, color=INK2)
    title(ax, heading)
    ax.set_xlabel("Training step")
    ax.set_xlim(0, 70)
    ax.set_ylim(*ylim)
    style(ax)


def logged(hist, key):
    return [(e["step"], e[key]) for e in hist if key in e and "loss" in e]


def training_curves():
    hist = load("run2/training_log_history.json")
    loss, acc = logged(hist, "loss"), logged(hist, "rewards/accuracies")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
    line_panel(a, [s for s, _ in loss], [v for _, v in loss], "Training loss (DPO plus SFT term)",
               lambda v: f"{v:.2f}", (0, 1), start_right=True)
    line_panel(b, [s for s, _ in acc], [v * 100 for _, v in acc], "Pairs with a higher reward for the chosen answer",
               lambda v: f"{v:.0f}%", (0, 100))
    b.yaxis.set_major_formatter(PCT)
    footer(fig, "Source: this project's Kaggle training run 2 on 2 T4 GPUs (2,000 pairs, 22 Oct 2026 UTC), logged every 5 steps.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(DOCS / "training_curves.png", facecolor=SURFACE)
    plt.close(fig)


def chosen_reward():
    """Run 1 against run 2: the implicit reward of the chosen answers, step by step."""
    runs = [("Run 1: DPO only", load("run1/training_log_history.json"), MUTED),
            ("Run 2: DPO plus SFT term", load("run2/training_log_history.json"), BLUE)]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    ax.axhline(0, color=INK2, linewidth=1)
    for label, hist, color in runs:
        pts = logged(hist, "rewards/chosen")
        xs, ys = [s for s, _ in pts], [v for _, v in pts]
        ax.plot(xs, ys, color=color, linewidth=2, solid_joinstyle="round")
        ax.scatter([xs[-1]], [ys[-1]], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{ys[-1]:+.2f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_title("Reward of the chosen answers during training", loc="left", fontsize=11, fontweight="bold",
                 color=INK, pad=30)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Implicit reward (log-ratio × beta)")
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    style(ax)
    handles = [plt.Line2D([0], [0], color=c, linewidth=2) for _, _, c in runs]
    ax.legend(handles, [lab for lab, _, _ in runs], frameon=False, loc="upper left", ncol=2,
              bbox_to_anchor=(0, 1.14), fontsize=9.5)
    footer(fig, "Source: this project's Kaggle training runs 1 and 2. A reward below zero means the chosen answers became less likely than under the base model.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(DOCS / "chosen_reward.png", facecolor=SURFACE)
    plt.close(fig)


def rounded_bar(ax, x, width, height, color):
    """Bar with a rounded data end and a square baseline."""
    r = min(width * 0.18, height * 0.5)
    ax.add_patch(FancyBboxPatch((x, 0), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=color, edgecolor="none", mutation_aspect=1 / 25))
    ax.add_patch(Rectangle((x, 0), width, min(height, 3), facecolor=color, edgecolor="none"))


def grouped(ax, groups, values, colors, width=0.3, gap=0.04):
    for i, _ in enumerate(groups):
        for j, (vals, color) in enumerate(zip(values, colors)):
            v = vals[i]
            x = i - (len(values) * width + (len(values) - 1) * gap) / 2 + j * (width + gap)
            rounded_bar(ax, x, width, v, color)
            ax.annotate(f"{v:.1f}%", (x + width / 2, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(groups)), groups, fontsize=10, color=INK)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    style(ax)


def legend_above(ax, labels, colors):
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c in colors]
    ax.legend(handles, labels, frameon=False, loc="upper left", ncol=len(labels), bbox_to_anchor=(0, 1.14), fontsize=10)


def results_chart():
    res = load("run2/results.json")
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
    metrics = [("valid_json_rate", "Valid JSON"), ("accuracy", "Accuracy")]
    grouped(ax, [m for _, m in metrics], [[res[k][m] * 100 for m, _ in metrics] for k in ("base", "dpo")], [BLUE, ORANGE])
    legend_above(ax, ["Base model", "After DPO plus SFT term"], [BLUE, ORANGE])
    ax.set_title("500 held-out headlines, scored with a strict JSON checker", loc="left", fontsize=11,
                 fontweight="bold", color=INK, pad=30)
    footer(fig, "Source: this project's Kaggle serving run 2 on vLLM (Qwen2.5-0.5B-Instruct before and after fine-tuning).")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(DOCS / "results.png", facecolor=SURFACE)
    plt.close(fig)


def per_label():
    out = {m: load(f"run2/outputs_{m}.json") for m in ("base", "dpo")}
    gold = out["dpo"]["gold"]
    vals, groups = [], [lab.capitalize() + chr(10) + f"{sum(g == lab for g in gold)} headlines" for lab in LABELS]
    for m in ("base", "dpo"):
        rows = [check(o, g) for o, g in zip(out[m]["outputs"], gold)]
        vals.append([100 * sum(r["correct"] for r, g in zip(rows, gold) if g == lab) / sum(g == lab for g in gold)
                     for lab in LABELS])
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    grouped(ax, groups, vals, [BLUE, ORANGE], width=0.32)
    legend_above(ax, ["Base model", "After DPO plus SFT term"], [BLUE, ORANGE])
    ax.set_title("Accuracy by true label", loc="left", fontsize=11,
                 fontweight="bold", color=INK, pad=30)
    footer(fig, "Source: this project's Kaggle serving run 2, 500 held-out headlines.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(DOCS / "accuracy_by_label.png", facecolor=SURFACE)
    plt.close(fig)


def baselines():
    out = load("run2/outputs_dpo.json")
    gold, texts = out["gold"], out["texts"]
    res = load("run2/results.json")
    neutral = 100 * sum(g == "neutral" for g in gold) / len(gold)
    words = 100 * sum(rules_label(t) == g for t, g in zip(texts, gold)) / len(gold)
    rows = [("Base model", res["base"]["accuracy"] * 100, BLUE), ("Always answer neutral", neutral, MUTED),
            ("Word-list classifier", words, MUTED), ("After DPO plus SFT term", res["dpo"]["accuracy"] * 100, ORANGE)]
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=200)
    for i, (label, v, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.5, color=color)
        ax.annotate(f"{v:.1f}%", (v, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=10, color=INK)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PCT)
    style(ax, axis="x")
    title(ax, "Accuracy against simple baselines")
    footer(fig, "Source: this project's Kaggle serving run 2, 500 held-out headlines (65.2% of them neutral).")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(DOCS / "accuracy_baselines.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    training_curves()
    chosen_reward()
    results_chart()
    per_label()
    baselines()
    print("figures written to", DOCS)
