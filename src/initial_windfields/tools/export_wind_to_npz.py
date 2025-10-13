#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export any initial windfield (static or not) to a flat-key NPZ.

Usage examples (run from repo root so that `src` is importable):
  python src/initial_windfields/tools/export_wind_to_npz.py \
    --name simple_static --out winds_export --write-seeds "0,1,2"

  python src/initial_windfields/tools/export_wind_to_npz.py \
    --name training_1 --out winds_export

  # Multiple names
  python src/initial_windfields/tools/export_wind_to_npz.py \
    --name training_1 --name training_2 --out winds_export/public

Notes
- We do NOT invent parameters. We just flatten the config your module exposes:
    grid_size, wind_init_params, wind_evol_params, (optional) static_wind
- You can override static detection via --static (true/false) if needed.
"""

import os
import argparse
import importlib
from typing import Any, Dict, Tuple, Optional, List

import numpy as np


# -------------------- helpers --------------------

def _to_bool(val: Any) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, np.integer)):
        return bool(int(val))
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def _detect_static(cfg: Dict[str, Any]) -> Optional[bool]:
    """Best-effort static detection from cfg if a 'static_wind' flag isn't present."""
    # 1) explicit flag
    if "static_wind" in cfg:
        return _to_bool(cfg["static_wind"])

    wep = cfg.get("wind_evol_params")
    if not isinstance(wep, dict):
        return None

    # 2) heuristic: all evolution magnitudes ≈ 0 -> static
    keys_zero = [
        "wind_change_prob", "perturbation_angle_amplitude",
        "perturbation_strength_amplitude", "rotation_bias", "bias_strength"
    ]
    try:
        return all(float(wep.get(k, 0.0)) == 0.0 for k in keys_zero)
    except Exception:
        return None


def _flatten_params(
    wip: Optional[Dict[str, Any]],
    wep: Optional[Dict[str, Any]],
    grid_size: Optional[Tuple[int, int]],
    static_wind: Optional[bool],
) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    # grid
    if grid_size is not None:
        d["grid_w"], d["grid_h"] = int(grid_size[0]), int(grid_size[1])
    # static flag
    if static_wind is not None:
        d["static_wind"] = int(1 if static_wind else 0)
    # init
    if isinstance(wip, dict):
        if "base_speed" in wip:
            d["wind_init_base_speed"] = float(wip["base_speed"])
        if "base_direction" in wip:
            bx, by = wip["base_direction"]
            d["wind_init_base_dir_x"] = float(bx)
            d["wind_init_base_dir_y"] = float(by)
        if "pattern_scale" in wip:
            d["wind_init_pattern_scale"] = int(wip["pattern_scale"])
        if "pattern_strength" in wip:
            d["wind_init_pattern_strength"] = float(wip["pattern_strength"])
        if "strength_variation" in wip:
            d["wind_init_strength_variation"] = float(wip["strength_variation"])
        if "noise" in wip:
            d["wind_init_noise"] = float(wip["noise"])
    # evol
    if isinstance(wep, dict):
        if "wind_change_prob" in wep:
            d["wind_evol_wind_change_prob"] = float(wep["wind_change_prob"])
        if "pattern_scale" in wep:
            d["wind_evol_pattern_scale"] = int(wep["pattern_scale"])
        if "perturbation_angle_amplitude" in wep:
            d["wind_evol_perturbation_angle_amplitude"] = float(wep["perturbation_angle_amplitude"])
        if "perturbation_strength_amplitude" in wep:
            d["wind_evol_perturbation_strength_amplitude"] = float(wep["perturbation_strength_amplitude"])
        if "rotation_bias" in wep:
            d["wind_evol_rotation_bias"] = float(wep["rotation_bias"])
        if "bias_strength" in wep:
            d["wind_evol_bias_strength"] = float(wep["bias_strength"])
    return d


def _save_npz(path: str, flat: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # ensure scalars are arrays to avoid object pickling
    payload = {k: np.asarray(v) for k, v in flat.items()}
    np.savez(path, **payload)
    print("Saved:", path)


def _write_seeds_txt(folder: str, seeds_str: str) -> None:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "seeds.txt"), "w", encoding="utf-8") as f:
        if "," in seeds_str:
            parts = [s for s in seeds_str.split(",") if s.strip() != ""]
        else:
            parts = [seeds_str.strip()] if seeds_str.strip() else []
        for p in parts:
            f.write(f"{int(p)}\n")
    print("Wrote seeds.txt in", folder)


# -------------------- export logic --------------------

def load_initial_wind_config(name: str) -> Dict[str, Any]:
    """Resolve a config by name via src.initial_windfields.__init__"""
    init_mod = importlib.import_module("src.initial_windfields.__init__")

    cfg = None
    getter = getattr(init_mod, "get_initial_windfield", None)
    if callable(getter):
        try:
            cfg = getter(name)
        except Exception:
            cfg = None
    if cfg is None and hasattr(init_mod, name):
        cfg = getattr(init_mod, name)

    if cfg is None:
        raise RuntimeError(f"Windfield '{name}' not found in src.initial_windfields.__init__")

    # Normalize to dict
    if isinstance(cfg, dict):
        return cfg
    # object with attributes
    out = {}
    for key in ("grid_size", "wind_init_params", "wind_evol_params", "static_wind"):
        if hasattr(cfg, key):
            out[key] = getattr(cfg, key)
    return out


def export_one(name: str, out_dir: str, static_override: Optional[bool], write_seeds: Optional[str]) -> str:
    cfg = load_initial_wind_config(name)

    grid_size = cfg.get("grid_size")
    wip = cfg.get("wind_init_params")
    wep = cfg.get("wind_evol_params")

    # static detection (cfg flag → heuristic) then CLI override
    static_flag = cfg.get("static_wind")
    static_auto = _detect_static(cfg) if static_flag is None else _to_bool(static_flag)
    static_final = static_override if static_override is not None else static_auto

    flat = _flatten_params(wip, wep, grid_size=grid_size, static_wind=static_final)

    npz_path = os.path.join(out_dir, f"{name}.npz")
    _save_npz(npz_path, flat)

    if write_seeds is not None:
        _write_seeds_txt(out_dir, write_seeds)

    return npz_path


# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True,
                    help="Windfield name in src.initial_windfields.__init__ (repeatable)")
    ap.add_argument("--out", type=str, default="winds_export",
                    help="Output directory for .npz files")
    ap.add_argument("--static", type=str, default=None,
                    help="Force static flag: true/false (optional override)")
    ap.add_argument("--write-seeds", type=str, default=None,
                    help='If set, write seeds.txt with comma-separated values, e.g. "0,1,2"')
    return ap.parse_args()


def main():
    args = parse_args()
    static_override = _to_bool(args.static) if args.static is not None else None

    exported: List[str] = []
    for nm in args.name:
        path = export_one(nm, args.out, static_override, args.write_seeds)
        exported.append(path)

    print("\nExported NPZ files:")
    for p in exported:
        print(" -", p)


if __name__ == "__main__":
    main()