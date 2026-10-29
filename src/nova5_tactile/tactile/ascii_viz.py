"""Console heat-maps for taxel arrays (handy in headless runs)."""

from __future__ import annotations

import numpy as np

_RAMP = " .:-=+*#%@"


def heatmap_lines(values: np.ndarray, vmax: float | None = None) -> list[str]:
    """Render a 2-D array as lines of ramp characters (row 0 printed last,
    so the fingertip end - highest row index - is at the top)."""
    v = np.asarray(values, dtype=float)
    top = float(vmax) if vmax and vmax > 0 else float(v.max(initial=0.0))
    if top <= 0:
        idx = np.zeros(v.shape, dtype=int)
    else:
        idx = np.clip((v / top) * (len(_RAMP) - 1), 0, len(_RAMP) - 1).round().astype(int)
    lines = ["".join(_RAMP[i] * 2 for i in row) for row in idx]
    return lines[::-1]


def side_by_side(frames: dict, vmax: float | None = None, gap: int = 4) -> str:
    """``frames`` maps a label to a TactileFrame (or a 2-D array)."""
    blocks, labels = [], []
    for label, fr in frames.items():
        arr = getattr(fr, "normal", fr)
        blocks.append(heatmap_lines(arr, vmax))
        total = float(np.asarray(arr).sum())
        labels.append(f"{label}: {total:6.2f} N")
    width = [max((len(b[0]) if b else 0) + 2, len(lbl)) for b, lbl in zip(blocks, labels)]
    height = max(len(b) for b in blocks) if blocks else 0
    sep = " " * gap
    out = [sep.join(lbl.ljust(w) for lbl, w in zip(labels, width))]
    for i in range(height):
        out.append(sep.join(("|" + (b[i] if i < len(b) else "") + "|").ljust(w) for b, w in zip(blocks, width)))
    return "\n".join(out)
