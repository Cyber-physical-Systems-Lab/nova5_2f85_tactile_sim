"""Grade the tactile validation checklist from finished runs (no Isaac needed).

Checks (from the nova5_2f85_tactile_sim scaffold's docs/tactile_model.md):
  1 static weight     contact total = m g within 3 % (TacSL: ratio reported, uncalibrated)
  2 location          tip load peaks in the last column, side load in the last row
  3 symmetric squeeze left/right totals within 10 % while holding; contact total = pad net force within 5 %
  4 shear sign        while holding, shear on both pads points to the fingertip (+y), ~ m g / 2 each
  5 noise floor       open gripper and empty pad read 0 everywhere

Usage: python scripts/validation_report.py --static runs/S --grasp runs/G --open runs/O [--md docs/validation.md]
"""
import argparse
import json
from pathlib import Path
import numpy as np

G = 9.81
OBJECT_MASS = .05  # grasp_demo.py box
HOLD = (10., 13.5)


def load(run, name):
    return json.loads((Path(run)/name).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--static', required=True); p.add_argument('--grasp', required=True); p.add_argument('--open', required=True)
    p.add_argument('--md')
    a = p.parse_args()
    st = load(a.static, 'static.json'); gr = load(a.grasp, 'result.json')['records']; op = load(a.open, 'result.json')['records']
    rows = []
    def add(check, backend, value, ok, note=''):
        rows.append((check, backend, value, 'info' if ok is None else 'PASS' if ok else 'FAIL', note))
    cases = st['cases']
    for name, exp in st['expected'].items():
        c = cases[name]['contact']['total_normal']
        add('1 static weight', 'contact', f'{name}: {c:.4f} N vs {exp["normal_n"]:.4f}', abs(c/exp['normal_n']-1) < .03)
        t = cases[name]['tacsl']['total_normal'] if 'tacsl' in cases[name] else None
        if t is not None:
            add('1 static weight', 'tacsl', f'{name}: {t:.2e} ({t/exp["normal_n"]:.2e} x m g)', None, 'penalty units, not N')
    for backend in ('contact', 'tacsl'):
        if backend not in cases['tip']:
            continue
        shape = np.shape(cases['tip'][backend]['normal'])
        tip, side = cases['tip'][backend]['peak_rc'], cases['side'][backend]['peak_rc']
        add('2 location', backend, f'tip peak {tip}, side peak {side} (grid {shape[0]}x{shape[1]})',
            bool(tip and side and tip[1] == shape[1]-1 and side[0] == shape[0]-1))
    hold = [r for r in gr if HOLD[0] <= r['t'] <= HOLD[1]]
    for key, backend in (('contact_taxels', 'contact'), ('tacsl', 'tacsl')):
        L = np.mean([r[key]['left']['normal_sum'] for r in hold]); R = np.mean([r[key]['right']['normal_sum'] for r in hold])
        add('3 symmetric squeeze', backend, f'L {L:.4g}, R {R:.4g}, diff {abs(L-R)/max((L+R)/2, 1e-12):.1%}',
            abs(L-R)/max((L+R)/2, 1e-12) < .10 if L+R > 0 else False)
    for side in ('left', 'right'):
        c = np.mean([r['contact_taxels'][side]['normal_sum'] for r in hold]); n = np.mean([r['pad_contact_force_n'][side] for r in hold])
        add('3 symmetric squeeze', 'contact', f'{side}: taxel sum {c:.3f} N vs pad net {n:.3f} N', abs(c/n-1) < .05 if n > 0 else False)
    # Static friction splits the weight between the pads arbitrarily, so test the sum along +y (to the fingertips).
    for key, backend in (('contact_taxels', 'contact'), ('tacsl', 'tacsl')):
        sh = {s: np.mean([r[key][s]['shear_sum'] for r in hold], 0) for s in ('left', 'right')}
        y = sum(v[1] for v in sh.values())
        txt = ', '.join(f'{s} ({v[0]:+.3g}, {v[1]:+.3g})' for s, v in sh.items())+f'; sum y {y:+.4g}'
        if backend == 'contact':
            add('4 shear sign', backend, txt, all(v[1] >= -.01 for v in sh.values()) and abs(y/(OBJECT_MASS*G)-1) < .05,
                f'sum y should be +m g = {OBJECT_MASS*G:.4f} N')
        else:
            add('4 shear sign', backend, txt, y > 0, 'velocity-based friction model, penalty units')
    for key, backend in (('contact_taxels', 'contact'), ('tacsl', 'tacsl')):
        worst = max(abs(r[key][s]['normal_sum']) for r in op for s in ('left', 'right'))
        empty = cases['empty'].get(backend, {}).get('total_normal', 0.)
        add('5 noise floor', backend, f'open run max {worst:.3g}, empty pad {empty:.3g}', worst == 0 and empty == 0)
    width = max(len(r[2]) for r in rows)
    lines = ['| Check | Backend | Result | Value | Note |', '|---|---|---|---|---|']
    lines += [f'| {c} | {b} | {s} | {v} | {n} |' for c, b, v, s, n in rows]
    for c, b, v, s, n in rows:
        print(f'{s:4}  {c:20} {b:8} {v:{width}}  {n}')
    if a.md:
        Path(a.md).write_text('\n'.join(lines)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
