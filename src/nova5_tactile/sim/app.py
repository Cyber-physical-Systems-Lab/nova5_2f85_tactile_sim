"""SimulationApp launch + asset resolution (Isaac Sim 6.x; 5.x fallbacks noted).

Import this module *before* anything from ``omni``/``isaacsim``/``pxr`` in a
standalone script: :func:`launch` must run first.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def launch(headless: bool, extra: dict | None = None):
    from isaacsim import SimulationApp  # noqa: PLC0415 - must be the first Isaac import

    cfg = {"headless": bool(headless)}
    cfg.update(extra or {})
    return SimulationApp(cfg)


def enable_extension(name: str) -> bool:
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    if mgr.is_extension_enabled(name):
        return True
    ok = mgr.set_extension_enabled_immediate(name, True)
    if not ok:
        log.warning("could not enable extension %s", name)
    return bool(ok)


def assets_root() -> str:
    """Isaac Sim assets root URL (S3 / Nucleus / local mirror)."""
    try:
        from isaacsim.storage.native import get_assets_root_path  # 5.x / 6.x
    except ImportError:  # pragma: no cover - very old releases
        from isaacsim.core.utils.nucleus import get_assets_root_path
    root = get_assets_root_path()
    if not root:
        raise RuntimeError("could not resolve the Isaac Sim assets root (check network / asset mirror settings)")
    return root.rstrip("/")


def url_exists(url: str) -> bool:
    import omni.client

    result, _entry = omni.client.stat(url)
    return result == omni.client.Result.OK


def resolve_gripper_usd(cfg) -> str:
    """Explicit ``paths.gripper_usd`` or the first existing candidate under the assets root."""
    explicit = cfg.paths.gripper_usd.strip()
    if explicit:
        if "://" in explicit:
            return explicit
        p = cfg.paths.resolve(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return str(p)
    root = assets_root()
    tried = []
    for cand in cfg.paths.gripper_usd_candidates:
        url = cand if "://" in cand else f"{root}/{cand.lstrip('/')}"
        tried.append(url)
        if url_exists(url):
            log.info("gripper USD: %s", url)
            return url
    raise FileNotFoundError(
        "no Robotiq 2F-85 USD found. Tried:\n  " + "\n  ".join(tried)
        + "\nSet paths.gripper_usd (or add a candidate) after locating it in the Content Browser."
    )
