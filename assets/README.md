# assets/

Everything here is generated and git-ignored by default:

| path | produced by |
|---|---|
| `urdf/nova5.urdf` | `scripts/fetch_nova5_description.sh` / `scripts/prepare_urdf.py` (absolute mesh paths → machine-specific) |
| `usd/nova5/` | `scripts/import_nova5.py` (Isaac Sim URDF importer output + `import_result.json`) |
| `usd/nova5_2f85_tactile.usd` | `scripts/assemble_robot.py` (references the two above; the Robotiq asset is referenced by URL) |
| `usd/nova5_2f85_tactile.meta.json` | `scripts/assemble_robot.py` (resolved prim paths, pad frames, warnings) |

If you hand-edit an asset and want to version it, move it out of the ignored
paths (e.g. `assets/usd_src/`) and commit it with Git LFS (see `.gitattributes`).
