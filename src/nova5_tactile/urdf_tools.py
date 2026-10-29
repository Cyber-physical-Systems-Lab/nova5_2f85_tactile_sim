"""Prepare a vendor URDF for the Isaac Sim URDF importer (stdlib only).

* expands ``.xacro`` files with the ``xacro`` tool when available,
* rewrites ``package://<pkg>/...`` and ``file://`` mesh URIs to absolute
  paths, by locating ``package.xml`` files under one or more search roots,
* reports meshes that cannot be found.

This avoids depending on a sourced ROS workspace inside Isaac Sim.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

PKG_RE = re.compile(r"package://([^/\"'\s]+)/([^\"'\s<>]+)")
FILE_RE = re.compile(r"file://(/[^\"'\s<>]+)")


@dataclass
class PrepareReport:
    output: Path
    packages: dict = field(default_factory=dict)
    unresolved_packages: set = field(default_factory=set)
    missing_files: list = field(default_factory=list)
    links: list = field(default_factory=list)
    joints: list = field(default_factory=list)


def find_packages(search_roots) -> dict[str, Path]:
    """Map ROS package name -> directory, from ``package.xml`` files."""
    found: dict[str, Path] = {}
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for pxml in root.rglob("package.xml"):
            try:
                name = ET.parse(pxml).getroot().findtext("name")
            except ET.ParseError:
                continue
            if name and name.strip() not in found:
                found[name.strip()] = pxml.parent.resolve()
    return found


def expand_xacro(path: Path, args: list[str] | None = None) -> str:
    cmd = None
    if shutil.which("xacro"):
        cmd = ["xacro", str(path)]
    else:
        try:
            import xacro  # noqa: F401

            cmd = [sys.executable, "-m", "xacro", str(path)]
        except ImportError:
            pass
    if cmd is None:
        raise RuntimeError("xacro not found: install it (pip install xacro, or source ROS 2) or pass a .urdf")
    res = subprocess.run(cmd + list(args or []), check=True, capture_output=True, text=True)
    return res.stdout


def rewrite_uris(text: str, packages: dict[str, Path], report: PrepareReport) -> str:
    def pkg_sub(m):
        pkg, rest = m.group(1), m.group(2)
        if pkg not in packages:
            report.unresolved_packages.add(pkg)
            return m.group(0)
        report.packages[pkg] = str(packages[pkg])
        p = packages[pkg] / rest
        if not p.exists():
            report.missing_files.append(str(p))
        return str(p)

    def file_sub(m):
        p = Path(m.group(1))
        if not p.exists():
            report.missing_files.append(str(p))
        return str(p)

    return FILE_RE.sub(file_sub, PKG_RE.sub(pkg_sub, text))


def prepare_urdf(src: Path, out: Path, search_roots, xacro_args=None) -> PrepareReport:
    src, out = Path(src), Path(out)
    text = expand_xacro(src, xacro_args) if src.name.endswith(".xacro") else src.read_text()
    report = PrepareReport(output=out)
    roots = list(search_roots) + [src.parent]
    text = rewrite_uris(text, find_packages(roots), report)
    root = ET.fromstring(text)
    if root.tag != "robot":
        raise ValueError(f"{src} does not look like a URDF (root tag {root.tag!r})")
    report.links = [e.get("name") for e in root.findall("link")]
    report.joints = [(e.get("name"), e.get("type")) for e in root.findall("joint")]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return report
