"""ManipulationNet Peg-in-Hole geometry from the CAD (Isaac-free)."""
from functools import lru_cache
import numpy as np
import pytest
from nova5_tactile.mnet import peg_in_hole as pih
from nova5_tactile import gripper_kinematics as gk


@lru_cache(None)
def holes():
    return pih.board_holes()


def test_twenty_holes_on_benchmark_grid():
    h = holes()
    assert sorted(h) == sorted(f'{s}{l}' for s in 'ABCDE' for l in '1234')
    for name, hole in h.items():
        assert abs(hole.center[1]*100-pih.ROW_Y_CM[hole.level]) < 1.


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_measured_gap_matches_clearance(level):
    # Per-side gap: about half the diametral clearance for levels 1-2, about the nominal value for 3-4 (CAD).
    for s in 'ABCD':
        gap = holes()[f'{s}{level}'].fit_gap_mm
        lo, hi = {1: (1.2, 1.6), 2: (.45, .55), 3: (.09, .12), 4: (.01, .025)}[level]
        assert lo < gap < hi, (s, level, gap)


def test_outside_pieces_are_convex_and_cover_tile():
    hole = holes()['E2']  # L-shaped, non-convex hole
    half = np.array([.03, .03])
    pieces = pih.outside_pieces(hole.polygon, hole.center, half)
    for p in pieces:
        assert pih._is_convex(pih._dedupe(p)) and pih.signed_area(pih._dedupe(p)) > 0
    area = sum(pih.signed_area(pih._dedupe(p)) for p in pieces)
    assert np.isclose(area, 4*half.prod()-pih.signed_area(hole.polygon), rtol=1e-6)


@pytest.mark.parametrize('shape', list('ABCDE'))
def test_peg_pieces_cover_section(shape):
    peg = pih.load_peg(shape)
    pieces = pih.inside_pieces(peg.polygon)
    assert np.isclose(sum(abs(pih.signed_area(p)) for p in pieces), pih.signed_area(peg.polygon), rtol=1e-6)
    assert peg.tris[..., 2].min() == pytest.approx(0) and peg.tris[..., 2].max() == pytest.approx(pih.PEG_LENGTH)


def test_gripper_gap_monotonic_and_inverts():
    qs = np.linspace(0, gk.Q_MAX, 20); gaps = [gk.pad_gap(q) for q in qs]
    assert np.all(np.diff(gaps) < 0) and gaps[0] > .07
    assert gk.pad_gap(gk.q_for_gap(.012)) == pytest.approx(.012, abs=2e-4)


def test_every_peg_fits_its_hole():
    for name, hole in holes().items():
        assert hole.min_gap_mm > .01, (name, hole.min_gap_mm)
        assert hole.min_gap_mm <= hole.fit_gap_mm+1e-9
