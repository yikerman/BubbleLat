"""Headline figure: reconstruction error against readout probe score, one point per model.

Reads the wandb runs that hold both test/* and probe/* (written by `evaluate` and `fit_test` in
scripts/common.sh), averages seeds, and writes figures/scatter.png plus a table on stdout.

    uv run python scripts/scatter.py
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.stats import spearmanr

FIELDS = ["dfun", "temperature", "velx", "vely"]
PROBES = ["vapor_frac", "interface_len", "bubble_count", "wall_temp"]
# family -> (label, color, marker, filled). At most three hues in a scatter (any two marks can
# be neighbors), so the ConvAE variants share its hue and differ by marker.
FAMILIES = {
    "downsample": ("Downsample", "#2a78d6", "s", True),
    "pod": ("POD", "#eb6834", "^", True),
    "convae": ("ConvAE", "#1baf7a", "o", True),
    "convvae": ("ConvVAE", "#1baf7a", "D", True),
    "conv_ae_random": ("ConvAE, untrained", "#1baf7a", "o", False),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def load() -> list[dict]:
    """One row per (experiment, model): seed mean and std of both scores, latest job only."""
    api = wandb.Api()
    seeds = defaultdict(lambda: defaultdict(list))  # (group, name) -> job -> runs
    for run in api.runs(f"{api.default_entity}/bubblelat"):
        s = run.summary
        # Runs trained before the cosine learning-rate decay are tagged "constant-lr" in wandb.
        if (run.state == "finished" and "probe/vapor_frac" in s and "test/vrmse/dfun" in s
                and "constant-lr" not in run.tags):
            seeds[run.group, run.name][run.id.split("-")[1]].append(run)  # "<exp>-<job>-<run>"
    rows = []
    for (group, name), jobs in seeds.items():
        runs = jobs[max(jobs, key=int)]
        err = np.array([np.mean([r.summary[f"test/vrmse/{f}"] for f in FIELDS]) for r in runs])
        probe = np.array([np.mean([r.summary[f"probe/{p}"] for p in PROBES]) for r in runs])
        family = group if group in FAMILIES else name.split("-")[0]
        beta = runs[0].config["model"]["loss"]["init_args"].get("beta") if family == "convvae" else None
        rows.append({"family": family, "name": name, "seeds": len(runs), "beta": beta,
                     "compression": round(runs[0].config["model/compression"]),
                     "err": err.mean(), "err_std": err.std(), "probe": probe.mean(), "probe_std": probe.std()})
    return sorted(rows, key=lambda r: r["err"])


def main() -> None:
    rows = load()
    print(f"{'model':20s} {'compr':>6s} {'seeds':>5s} {'VRMSE':>7s} {'probe R2':>9s}")
    for r in rows:
        label = FAMILIES[r["family"]][0] + (f" b={r['beta']}" if r["beta"] else "")
        print(f"{label:20s} {r['compression']:5d}x {r['seeds']:5d} {r['err']:7.3f} {r['probe']:9.3f}")
    trained = [r for r in rows if r["family"] != "conv_ae_random"]
    rho = spearmanr([r["err"] for r in trained], [r["probe"] for r in trained]).statistic
    print(f"Spearman(error, probe) over {len(trained)} trained or fitted models: {rho:.2f}")

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c3c2b7", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained", facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    seen = set()
    for r in rows:
        label, color, marker, filled = FAMILIES[r["family"]]
        ax.errorbar(r["err"], r["probe"], xerr=r["err_std"], yerr=r["probe_std"], fmt=marker,
                    ms=9, color=color, mfc=color if filled else SURFACE, mec=color, mew=2,
                    elinewidth=1, label=None if label in seen else label)
        seen.add(label)
        # ConvAE and its beta-VAE variants nearly coincide: label ConvAE on the left and stack
        # the beta labels on the right.
        if r["beta"]:
            note, offset, ha = f"β={r['beta']}", (9, 9 if r["beta"] >= 0.1 else -14), "left"
        else:
            # Points of one family that coincide share one label, e.g. "4×, 16×".
            twins = [q for q in rows if q["family"] == r["family"] and not q["beta"]
                     and abs(np.log(q["err"] / r["err"])) < 0.03 and abs(q["probe"] - r["probe"]) < 0.01]
            if r is not twins[0]:
                continue
            note = ", ".join(f"{c}×" for c in sorted(q["compression"] for q in twins))
            offset, ha = ((-7, 6), "right") if r["family"] == "convae" else ((7, 6), "left")
        ax.annotate(note, (r["err"], r["probe"]), xytext=offset, textcoords="offset points",
                    fontsize=9, color="#52514e", ha=ha)
    ax.set_xscale("log")
    ax.set_xticks([0.15, 0.2, 0.3, 0.5, 0.75, 1.0], ["0.15", "0.2", "0.3", "0.5", "0.75", "1.0"])
    ax.minorticks_off()
    ax.set_xlabel("reconstruction error: test VRMSE, mean of 4 fields (log scale, lower is better)")
    ax.set_ylabel("readout: ridge probe R², mean of 4 targets")
    ax.set_title(f"Probe score follows reconstruction only loosely (Spearman {rho:.2f}, {len(trained)} models)",
                 loc="left", fontsize=12)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    names = [f[0] for f in FAMILIES.values()]
    order = sorted(range(len(labels)), key=lambda i: names.index(labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, loc="lower left")
    fig.savefig("figures/scatter.png", dpi=200)
    print("wrote figures/scatter.png")


if __name__ == "__main__":
    main()
