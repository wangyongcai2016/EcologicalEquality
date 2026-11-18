#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 01 — Ecological Function Index (EFI)   [GeoTIFF; no-args / VSCode-run]
================================================================================
Paper section 2.2.1, Eq. 2 (deepened version).

    EFI_ij = 1/3 * min(FVC_ij/FVC_ref,1) + 1/3 * min(LAI_ij/LAI_ref,1)
           + 1/3 * min(GPP_ij/GPP_ref,1)

Reference value of a unit = mean of the top-10th-percentile pixels among the
land-cover-invariant ("stable") pixels of that unit; bootstrapped for a 95% CI.

HOW TO RUN (no command-line arguments):
    1. Open geo_io.py and set the paths in CONFIG (VEG_DIR, UNITS_TIF, ...).
    2. Just run this file in VSCode (Run / F5) or:  python 01_efi.py

INPUT (from geo_io.CONFIG):
    VEG_DIR/fvc_YYYY.tif, lai_YYYY.tif, gpp_YYYY.tif   annual, START..END_YEAR
    UNITS_TIF    int assessment-unit id (1..212); 0/NoData = masked
    STABLE_TIF   >0 = land-cover-invariant reference pixel
    (SINGLE_UNIT=True -> ignore UNITS_TIF, treat all grassland as one unit)

OUTPUT (to CONFIG['OUT_DIR']):
    GeoTIFF : efi_mean.tif, efi_ci_low.tif, efi_ci_high.tif, efi_lowconf.tif,
              efi_YYYY.tif (one per year)
    NumPy   : efi_stack.npy, efi_mean.npy, efi_ci_low/high.npy, efi_lowconf_mask.npy
    Table   : efi_summary.csv     Preview: 01_efi_preview.png
================================================================================
"""
from __future__ import annotations
import os, csv, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- method parameters ----
REF_PERCENTILE = 90.0
N_BOOTSTRAP    = 1000
MIN_REF_PIXELS = 30
RNG            = np.random.default_rng(42)


def unit_reference_bootstrap(param_mean, unit_labels, stable_mask):
    """Per-unit reference value (top-decile mean of stable pixels) + bootstrap CI."""
    ref, ref_lo, ref_hi, n_ref = {}, {}, {}, {}
    for u in np.unique(unit_labels):
        if u <= 0:
            continue
        sel = (unit_labels == u) & stable_mask & np.isfinite(param_mean)
        vals = param_mean[sel]
        if vals.size == 0:
            continue
        thr = np.percentile(vals, REF_PERCENTILE)
        top = vals[vals >= thr]
        if top.size == 0:
            top = vals
        n_ref[int(u)] = int(top.size)
        ref[int(u)] = float(top.mean())
        boot = np.array([RNG.choice(top, size=top.size, replace=True).mean()
                         for _ in range(N_BOOTSTRAP)])
        ref_lo[int(u)] = float(np.percentile(boot, 2.5))
        ref_hi[int(u)] = float(np.percentile(boot, 97.5))
    return ref, ref_lo, ref_hi, n_ref


def _ref_field(unit_labels, ref):
    out = np.full(unit_labels.shape, np.nan, "float32")
    for u, v in ref.items():
        out[unit_labels == u] = v
    return out


def capped_ratio(value, reference):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(value / reference, 0.0, 1.0)


def compute_efi(fvc, lai, gpp, unit_labels, refs):
    r1 = capped_ratio(fvc, _ref_field(unit_labels, refs["fvc"]))
    r2 = capped_ratio(lai, _ref_field(unit_labels, refs["lai"]))
    r3 = capped_ratio(gpp, _ref_field(unit_labels, refs["gpp"]))
    efi = (r1 + r2 + r3) / 3.0
    efi[unit_labels <= 0] = np.nan
    return efi


# ------------------------------------------------------------------ loaders
def load_geotiffs():
    c = CONFIG
    print(f"[EFI] reading annual GeoTIFFs from {c['VEG_DIR']} "
          f"({c['START_YEAR']}-{c['END_YEAR']})")
    fvc, prof, yrs = gio.stack_annual(c["VEG_DIR"][c["FVC"]], c["FVC_NAME"],
                                      c["START_YEAR"], c["END_YEAR"], c["VEG_PATTERN"])
    lai, _, _ = gio.stack_annual(c["VEG_DIR"][c["LAI"]], c["LAI_NAME"],
                                 c["START_YEAR"], c["END_YEAR"], c["VEG_PATTERN"])
    gpp, _, _ = gio.stack_annual(c["VEG_DIR"][c["GPP"]], c["GPP_NAME"],
                                 c["START_YEAR"], c["END_YEAR"], c["VEG_PATTERN"])
    if not (fvc.shape == lai.shape == gpp.shape):
        raise ValueError(f"[EFI] stacks differ: fvc{fvc.shape} lai{lai.shape} gpp{gpp.shape}")
    H, W = fvc.shape[1:]

    if c["SINGLE_UNIT"]:
        valid = np.isfinite(fvc).any(0) | np.isfinite(lai).any(0) | np.isfinite(gpp).any(0)
        unit_labels = valid.astype("int32")
        print("[EFI] SINGLE_UNIT: whole grassland treated as one unit")
    else:
        u_arr, _ = gio.read_tif(c["UNITS_TIF"])
        unit_labels = np.nan_to_num(u_arr, nan=0).astype("int32")
        if unit_labels.shape != (H, W):
            raise ValueError(f"[EFI] units {unit_labels.shape} != grid {(H,W)}")

    if gio.files_available(c["STABLE_TIF"]):
        s_arr, _ = gio.read_tif(c["STABLE_TIF"])
        stable = np.nan_to_num(s_arr, nan=0) > 0
    else:
        stable = unit_labels > 0
        print(f"[EFI] STABLE_TIF not found; using all unit pixels as references")
    return fvc, lai, gpp, unit_labels, stable, prof, yrs


def generate_demo():
    c = CONFIG
    H = W = 40
    yrs = list(range(c["START_YEAR"], c["END_YEAR"] + 1)); n = len(yrs)
    yy, xx = np.mgrid[0:H, 0:W]
    unit_labels = (1 + (xx // 10) + 4 * (yy // 10)).astype("int32")
    grass = np.ones((H, W), bool); grass[(xx - W + 3) ** 2 + (yy - 2) ** 2 < 6] = False
    unit_labels[~grass] = 0
    stable = grass & (RNG.random((H, W)) > 0.2)
    base = 0.4 + 0.3 * RNG.random((H, W))
    plateau = (xx >= 2 * W // 3) & (yy >= H // 2); trend = np.where(plateau, 0.0, 0.010)
    fvc = np.empty((n, H, W), "float32"); lai = np.empty_like(fvc); gpp = np.empty_like(fvc)
    for t in range(n):
        lvl = np.clip(base + trend * t + 0.03 * RNG.standard_normal((H, W)), 0.05, 0.95)
        fvc[t] = lvl; lai[t] = lvl * (2.5 + 0.5 * RNG.random((H, W)))
        gpp[t] = lvl * (900 + 200 * RNG.random((H, W)))
    fvc[:, ~grass] = np.nan; lai[:, ~grass] = np.nan; gpp[:, ~grass] = np.nan
    return fvc, lai, gpp, unit_labels, stable, None, yrs


# ------------------------------------------------------------------ main
def main():
    c = CONFIG
    out = gio.ensure_outdir()
    have = gio.files_available(c["VEG_DIR"]) and (c["SINGLE_UNIT"] or gio.files_available(c["UNITS_TIF"]))
    if have:
        fvc, lai, gpp, unit_labels, stable, prof, yrs = load_geotiffs()
    elif c["ALLOW_DEMO"]:
        print("[EFI] data paths not found; running synthetic demo "
              "(edit CONFIG in geo_io.py to use your GeoTIFFs)")
        fvc, lai, gpp, unit_labels, stable, prof, yrs = generate_demo()
    else:
        raise FileNotFoundError("[EFI] configured GeoTIFFs not found and ALLOW_DEMO=False")

    n = fvc.shape[0]; H, W = fvc.shape[1:]
    print(f"[EFI] grid {H}x{W}, years {yrs[0]}-{yrs[-1]} ({n}), "
          f"units {len(np.unique(unit_labels))-1}, stable px {int(stable.sum())}")

    refs, refs_lo, refs_hi, n_ref = {}, {}, {}, {}
    for name, arr in [("fvc", fvc), ("lai", lai), ("gpp", gpp)]:
        pm = np.nanmean(arr, 0)
        refs[name], refs_lo[name], refs_hi[name], n_ref[name] = \
            unit_reference_bootstrap(pm, unit_labels, stable)

    efi_stack = np.stack([compute_efi(fvc[t], lai[t], gpp[t], unit_labels, refs)
                          for t in range(n)], 0)
    efi_mean = np.nanmean(efi_stack, 0)
    efi_lo = np.nanmean(np.stack([compute_efi(fvc[t], lai[t], gpp[t], unit_labels,
        {"fvc": refs_hi["fvc"], "lai": refs_hi["lai"], "gpp": refs_hi["gpp"]})
        for t in range(n)], 0), 0)
    efi_hi = np.nanmean(np.stack([compute_efi(fvc[t], lai[t], gpp[t], unit_labels,
        {"fvc": refs_lo["fvc"], "lai": refs_lo["lai"], "gpp": refs_lo["gpp"]})
        for t in range(n)], 0), 0)

    lowconf = np.zeros(unit_labels.shape, bool)
    for name in ("fvc", "lai", "gpp"):
        for u, cnt in n_ref[name].items():
            if cnt < MIN_REF_PIXELS:
                lowconf[unit_labels == u] = True

    # NumPy (for downstream scripts)
    np.save(os.path.join(out, "efi_stack.npy"), efi_stack.astype(np.float32))
    np.save(os.path.join(out, "efi_mean.npy"), efi_mean.astype(np.float32))
    np.save(os.path.join(out, "efi_ci_low.npy"), efi_lo.astype(np.float32))
    np.save(os.path.join(out, "efi_ci_high.npy"), efi_hi.astype(np.float32))
    np.save(os.path.join(out, "efi_lowconf_mask.npy"), lowconf)

    # GeoTIFF
    if prof is not None:
        gio.write_tif(os.path.join(out, "efi_mean.tif"), efi_mean, prof)
        gio.write_tif(os.path.join(out, "efi_ci_low.tif"), efi_lo, prof)
        gio.write_tif(os.path.join(out, "efi_ci_high.tif"), efi_hi, prof)
        gio.write_tif(os.path.join(out, "efi_lowconf.tif"),
                      lowconf.astype("uint8"), prof, "uint8", 255)
        for t, yr in enumerate(yrs):
            gio.write_tif(os.path.join(out, f"efi_{yr}.tif"), efi_stack[t], prof)
        print(f"[EFI] wrote GeoTIFFs (efi_mean.tif, efi_ci_*.tif, efi_{yrs[0]}..{yrs[-1]}.tif)")
    else:
        print("[EFI] demo mode: wrote .npy only (no georeference)")

    with open(os.path.join(out, "efi_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["unit","param","ref","ref_ci_low","ref_ci_high","n_ref"])
        for name in ("fvc","lai","gpp"):
            for u in sorted(refs[name]):
                w.writerow([u, name, f"{refs[name][u]:.4f}",
                            f"{refs_lo[name][u]:.4f}", f"{refs_hi[name][u]:.4f}", n_ref[name][u]])

    print(f"[EFI] mean EFI = {np.nanmean(efi_mean):.3f} "
          f"[{np.nanmean(efi_lo):.3f}, {np.nanmean(efi_hi):.3f}]")
    vn = int(np.isfinite(efi_mean).sum())
    print(f"[EFI] low-confidence pixels: {int(lowconf.sum())} "
          f"({100*lowconf.sum()/max(vn,1):.1f}% of valid)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(5,4)); plt.imshow(efi_mean, cmap="YlGn", vmin=0, vmax=1)
        plt.colorbar(label=f"EFI ({yrs[0]}-{yrs[-1]} mean)"); plt.title("EFI")
        plt.tight_layout(); plt.savefig(os.path.join(out, "01_efi_preview.png"), dpi=120)
        print(f"[EFI] preview -> {out}/01_efi_preview.png")
    except Exception as e:
        print(f"[EFI] (preview skipped: {e})")
    print("[EFI] done.")


if __name__ == "__main__":
    main()
