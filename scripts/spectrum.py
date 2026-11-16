"""Reconstruction error by wavenumber: one panel per compression ratio, one line per model.

Each line is sqrt(error power / target power) in a wavenumber shell, averaged over the fields
(or the fields named), from the "spectrum" that the test step stores in each wandb run.
1.0 means that scale is not reconstructed at all.

    uv run python scripts/spectrum.py [field ...]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from bubblelat.runs import latest_runs

FIELDS = ["dfun", "temperature", "velx", "vely"]
# Three hues, and a line style each so the families stay apart without color.
FAMILIES = {"downsample": ("Downsample", "#2a78d6", "--"), "pod": ("POD", "#eb6834", "-."),
            "convae": ("ConvAE", "#1baf7a", "-")}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def main() -> None:
    fields = sys.argv[1:] or FIELDS
    curves = {}  # (compression, family) -> relative error per shell, mean of seeds and fields
    for (group, name), runs in latest_runs("spectrum.dfun.error", "test/vrmse/dfun").items():
        family = name.split("-")[0]
        if group == "conv_ae_random" or family not in FAMILIES:
            continue
        ratio = [np.sqrt(np.array(r.summary[f"spectrum.{f}.error"]) / np.array(r.summary[f"spectrum.{f}.target"]))
                 for r in runs for f in fields]
        curves[round(runs[0].config["model/compression"]), family] = np.mean(ratio, axis=0)

    ratios = sorted({c for c, _ in curves})
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, len(ratios), figsize=(3.1 * len(ratios), 3.4), sharey=True,
                             layout="constrained", facecolor=SURFACE, squeeze=False)
    k = np.arange(1, 65)  # shells up to the Nyquist wavenumber of a 128 px frame
    for ax, c in zip(axes[0], ratios):
        ax.set_facecolor(SURFACE)
        for family, (label, color, style) in FAMILIES.items():
            if (c, family) in curves:
                ax.plot(k, curves[c, family][1:65], style, color=color, linewidth=2, label=label)
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        ax.minorticks_off()
        ax.set_title(f"{c}× compression", fontsize=10, loc="left")
        ax.set_xlabel("wavenumber (cycles per domain)")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("relative error at that scale")
    axes[0, 0].set_yticks([0.01, 0.03, 0.1, 0.3, 1.0], ["0.01", "0.03", "0.1", "0.3", "1.0"])
    axes[0, -1].legend(frameon=False, loc="lower right")
    what = "mean of 4 fields" if fields == FIELDS else ", ".join(fields)
    fig.suptitle(f"Reconstruction error by scale ({what}); 1.0 = that scale is lost",
                 x=0.01, ha="left", fontsize=11)
    out = "figures/spectrum.png" if fields == FIELDS else f"figures/spectrum_{'_'.join(fields)}.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
