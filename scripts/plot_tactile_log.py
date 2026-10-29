#!/usr/bin/env python3
"""Plot a tactile log written by run_grasp_demo.py (plain Python + matplotlib).

    python3 scripts/plot_tactile_log.py logs/grasp_*.npz [--save out.png]
"""

import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--save", default=None, help="write a PNG instead of opening a window")
    a = ap.parse_args()
    try:
        import matplotlib

        if a.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: pip install matplotlib", file=sys.stderr)
        return 1

    z = np.load(a.log)
    spec = json.loads(str(z["spec"]))
    t = z["time"]
    pads = sorted({k.split("/")[0] for k in z.files if "/" in k})

    fig = plt.figure(figsize=(4 + 3 * len(pads), 7))
    ax = fig.add_subplot(2, 1, 1)
    for p in pads:
        ax.plot(t, z[f"{p}/total_normal"], label=f"{p} total normal")
    if "phase" in z.files:
        for k in np.flatnonzero(np.diff(z["phase"]) != 0):
            ax.axvline(t[k + 1], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force [N]")
    ax.legend()
    ax.grid(alpha=0.3)

    for i, p in enumerate(pads):
        n, s = z[f"{p}/normal"], z[f"{p}/shear"]
        k = int(np.argmax(z[f"{p}/total_normal"])) if len(n) else 0
        axh = fig.add_subplot(2, len(pads), len(pads) + i + 1)
        im = axh.imshow(n[k], origin="lower", cmap="inferno", aspect="auto")
        r, c = np.mgrid[0:n.shape[1], 0:n.shape[2]]
        if np.any(s[k]):  # (y, x) shear drawn on (col, row) axes
            axh.quiver(c, r, s[k][..., 1], s[k][..., 0], color="cyan")
        axh.set_title(f"{p} @ t={t[k]:.2f}s (rows=+x tip, cols=+y)")
        fig.colorbar(im, ax=axh, label="N")
    fig.suptitle(f"{spec['rows']}x{spec['cols']} taxels, {spec['size_x']*1e3:.0f}x{spec['size_y']*1e3:.0f} mm")
    fig.tight_layout()
    if a.save:
        fig.savefig(a.save, dpi=120)
        print(f"wrote {a.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
