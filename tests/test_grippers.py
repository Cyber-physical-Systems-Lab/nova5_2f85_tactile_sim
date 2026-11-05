"""Gripper registry and task naming (Isaac-free)."""
import pytest
from nova5_tactile.grippers import GRIPPERS, DEFAULT


def test_registry_consistent():
    assert DEFAULT == 'cps_tactile' and set(GRIPPERS) == {'cps_tactile', 'robotiq_2f85'}
    for g in GRIPPERS.values():
        assert g.driver in g.joints and g.joints[g.driver] == 1.
        assert set(g.fingers) == {'left', 'right'}
        assert g.effort_unit in ('N', 'N m') and 0 < g.grip_effort <= g.effort_limit
        # A held 50 mm peg must reach 25 mm (full board depth) below the lowest fingertip.
        assert .050-g.peg_top_above_contact-g.tip_below_contact >= .025


def test_cps_jaws_coupled_symmetrically():
    g = GRIPPERS['cps_tactile']
    assert g.effort_unit == 'N' and list(g.joints.values()) == [1., 1.]


def test_task_ids():
    pytest.importorskip('gymnasium')
    from nova5_tactile import tasks
    ids = {tasks.task_id(t, v) for t in tasks.TASKS for v in tasks.VARIANTS}
    assert len(ids) == 40
    assert tasks.task_id('A1') == 'Nova5-MNet-PegInHole-A1-v0'
    assert tasks.task_id('E4', '2F85') == 'Nova5-MNet-PegInHole-E4-2F85-v0'
