#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 08 — Function-Stability diagnostic space (H1)   [tiled, parallel | national natural ecosystems]
================================================================================
Paper 2.7 & Table 3 -- THE HEADLINE ANALYSIS. Every stable natural pixel is placed in (dEFI, dESI) and
classified into four regimes by the SIGN of the raw slopes:
    0 True recovery     dEFI>0, dESI>0
    1 Fragile greening  dEFI>0, dESI<=0   <-- productivity up, stability eroding (KEY)
    2 Low-but-stable    dEFI<=0, dESI>0
    3 Collapse          dEFI<=0, dESI<=0

dEFI = Theil-Sen slope of the annual EFI stack (efi_stack.npy, START_YEAR .. min(END_YEAR, LU_END_YEAR)).
dESI = esi_trend.npy from 02_esi.py (Theil-Sen slope of the moving-window ESI series).
Slopes are NOT standardised (the previous version z-scored them, which turned ">0" into "above the national
mean slope"); the sign keeps its absolute meaning.

Robustness / reporting (all from the same pass, written to diag_regime_shares.csv):
    scheme "all"              every analysed pixel
    scheme "excl_lowconf"     drop the highest-NPP tercile (esi_reliability == 0; Smith & Boers 2023)
    scheme "sigEFI"           keep pixels whose dEFI passes Mann-Kendall (p < 0.05)
    scheme "sigEFI_excl_lowconf"  both
  groups: ALL, each ecosystem type (forest / shrub / grassland / wetland / sparse), and each assessment unit
  (units can be dissolved into the three ecological regions afterwards: Fu et al. 2001).

Ecosystem type = ESA-CCI class of TYPE_YEAR (2010, mid-period) resampled (nearest) onto the analysis grid and
grouped by TYPE_GROUPS below; cached as eco_type.tif.

HOW TO RUN: 01_efi_new.py and 02_esi.py first (so that efi_stack.npy, esi_trend.npy, esi_reliability.tif exist).
            Pilot: same ROI / EQI_OUT_DIR switches as 02_esi.py.

INPUT  : efi_stack.npy (nY_total,H,W; read through a memmap, full UNITS_TIF grid), esi_trend.npy,
         esi_reliability.tif (optional), ESACCI-LC-<TYPE_YEAR>.tif, UNITS_TIF, STABLE_TIF
OUTPUT (CONFIG['OUT_DIR']): diag_dEFI.tif/.npy, diag_dESI.tif/.npy, diag_pEFI.tif (MK p), diag_regime.tif/.npy
        (uint8: 0-3, 255 = not analysed), eco_type.tif, diag_regime_shares.csv, 08_diagnostic_space.png
================================================================================
"""
from __future__ import annotations
import os, csv, time, warnings
import numpy as np
from scipy.special import ndtr
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(21)
REGIME_NAMES = ["True recovery", "Fragile greening", "Low-but-stable", "Collapse"]
TYPE_YEAR = 2010
# ESA-CCI codes -> 5 natural ecosystem types (see geo_io.ESACCI_NATURAL_CODES). Author decision; edit as needed.
TYPE_GROUPS = {
    1: ("Forest",    {50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100}),
    2: ("Shrub",     {120, 121, 122}),
    3: ("Grassland", {110, 130}),
    4: ("Wetland",   {160, 170, 180}),
    5: ("Sparse", {140, 150, 151, 152, 153}),          # sparse vegetation only; bare land (200-202) is never assessed
}
TYPE_NAMES = {k: v[0] for k, v in TYPE_GROUPS.items()}
ALPHA = 0.05
NOD = 255


# ------------------------------------------------------------ vectorised trend statistics (columns = pixels)
def theil_sen_mk(E):
    """E : (n, m). Returns Theil-Sen slope (m,) and two-sided Mann-Kendall p (m,) per column (NaN-aware)."""
    n, m = E.shape
    ii, jj = np.triu_indices(n, 1)
    D = E[jj] - E[ii]
    S = D / (jj - ii)[:, None]
    slope = np.nanmedian(S, axis=0)
    sg = np.sign(D)
    s_stat = np.nansum(sg, axis=0)
    nv = np.isfinite(E).sum(0).astype("float64")
    var = nv * (nv - 1) * (2 * nv + 5) / 18.0               # no tie correction (continuous data)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(s_stat > 0, (s_stat - 1) / np.sqrt(var), np.where(s_stat < 0, (s_stat + 1) / np.sqrt(var), 0.0))
    p = 2.0 * (1.0 - ndtr(np.abs(z)))
    bad = nv < 5
    slope[bad] = np.nan
    p[bad] = np.nan
    return slope.astype("float32"), p.astype("float32")


def classify(dEFI, dESI):
    """Raw-sign classification -> uint8 (0-3); NOD where either slope is NaN."""
    reg = np.full(dEFI.shape, NOD, "uint8")
    ok = np.isfinite(dEFI) & np.isfinite(dESI)
    up_f = dEFI > 0
    up_s = dESI > 0
    reg[ok & up_f & up_s] = 0
    reg[ok & up_f & ~up_s] = 1
    reg[ok & ~up_f & up_s] = 2
    reg[ok & ~up_f & ~up_s] = 3
    return reg


# ------------------------------------------------------------ ecosystem-type map
def build_type_map(grid, out):
    path = os.path.join(out, "eco_type.tif")
    if os.path.exists(path):
        import rasterio
        with rasterio.open(path) as s:
            if (s.height, s.width) == (grid["height"], grid["width"]):
                return path
    lu = os.path.join(CONFIG["LANDUSE_DIR"], CONFIG["LU_PATTERN"].format(var=CONFIG["LU_NAME"], year=TYPE_YEAR))
    if not os.path.exists(lu):
        import glob
        raw = sorted(glob.glob(os.path.join(CONFIG.get("LANDUSE_RAW_DIR", ""), f"*-{TYPE_YEAR}-*.tif")))
        if not raw:
            return None
        lu = raw[0]
        print(f"[DIAG] {CONFIG['LANDUSE_DIR']} has no {TYPE_YEAR} file; using the raw ESA-CCI {os.path.basename(lu)}")
    print(f"[DIAG] building ecosystem-type map from {os.path.basename(lu)} (nearest) ...")
    code = gio.resample_to_ref(lu, grid, categorical=True)
    tmap = np.zeros(code.shape, "uint8")
    for k, (_, codes) in TYPE_GROUPS.items():
        tmap[np.isin(code, list(codes))] = k
    gio.write_tif(path, tmap, grid, "uint8", 0)
    return path


# ------------------------------------------------------------ per-tile work
def process_tile(win_t, efi_path, nY, tpath, rpath, esi_trend_path):
    from rasterio.windows import Window
    import rasterio
    r0, c0, h, w = win_t
    win = Window(c0, r0, w, h)
    mask = gio.read_mask(win)
    if not mask.any():
        return None
    mw = gio._master_window(win)
    R0, C0 = int(mw.row_off), int(mw.col_off)
    efi = np.load(efi_path, mmap_mode="r")
    blk = np.asarray(efi[:nY, R0:R0 + h, C0:C0 + w], "float32")
    dESI = np.asarray(np.load(esi_trend_path, mmap_mode="r")[r0:r0 + h, c0:c0 + w], "float32")
    enough = np.isfinite(blk).sum(0) >= max(5, nY - 3)
    ok = mask & enough & np.isfinite(dESI)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok.ravel())
    dE_flat, p_flat = theil_sen_mk(blk.reshape(nY, -1)[:, idx])
    dEFI = np.full(h * w, np.nan, "float32"); dEFI[idx] = dE_flat
    pEFI = np.full(h * w, np.nan, "float32"); pEFI[idx] = p_flat
    dEFI = dEFI.reshape(h, w); pEFI = pEFI.reshape(h, w)
    reg = classify(dEFI, dESI)
    if tpath:
        with rasterio.open(tpath) as s:
            typ = s.read(1, window=win)
    else:
        typ = np.zeros((h, w), "uint8")
    if rpath and os.path.exists(rpath):
        with rasterio.open(rpath) as s:
            rel = np.nan_to_num(s.read(1, window=win), nan=1).astype("uint8")
    else:
        rel = np.ones((h, w), "uint8")                    # unknown reliability -> treat as "middle"
    units = gio.read_units(win)
    sig = np.isfinite(pEFI) & (pEFI < ALPHA)
    v = reg != NOD
    # counts: [type 0-5][reliab 0-2][sig 0-1][regime 0-3]
    key = ((typ.astype(np.int64) * 3 + rel) * 2 + sig) * 4 + reg.astype(np.int64)
    cnt = np.bincount(key[v], minlength=6 * 3 * 2 * 4).reshape(6, 3, 2, 4)
    # per unit (all / excl_lowconf / sig variants collapsed to counts by regime, reliab, sig)
    ucnt = {}
    for u in np.unique(units[v]):
        if u <= 0:
            continue
        sel = v & (units == u)
        k2 = (rel[sel].astype(np.int64) * 2 + sig[sel]) * 4 + reg[sel].astype(np.int64)
        ucnt[int(u)] = np.bincount(k2, minlength=3 * 2 * 4).reshape(3, 2, 4)
    return r0, c0, h, w, dEFI, pEFI, dESI, reg, cnt, ucnt


# ------------------------------------------------------------ summaries
def shares_rows(cnt, ucnt):
    """cnt (6,3,2,4), ucnt {unit: (3,2,4)} -> csv rows."""
    rows = []

    def pack(c, label):
        # c : (3 reliab, 2 sig, 4 regime)
        schemes = {
            "all": c.sum((0, 1)),
            "excl_lowconf": c[1:].sum((0, 1)),
            "sigEFI": c[:, 1].sum(0),
            "sigEFI_excl_lowconf": c[1:, 1].sum(0),
        }
        for s, v in schemes.items():
            n = int(v.sum())
            rows.append([label, s, n, *[f"{100 * x / n:.2f}" if n else "" for x in v]])

    pack(cnt.sum(0), "ALL")
    for t in range(0, 6):
        if cnt[t].sum() == 0:
            continue
        pack(cnt[t], f"type_{t}_{TYPE_NAMES.get(t, 'unknown')}")
    for u in sorted(ucnt):
        pack(ucnt[u], f"unit_{u}")
    return rows


def run_real(out):
    from joblib import Parallel, delayed
    c = CONFIG
    start = c["START_YEAR"]
    end = min(c["END_YEAR"], c.get("LU_END_YEAR", c["END_YEAR"]))
    nY = end - start + 1
    efi_path = os.path.join(out, "efi_stack.npy")
    if not os.path.exists(efi_path):
        alt = os.path.join(gio.PROJECT_ROOT, "output", "efi_stack.npy")
        if os.path.exists(alt):
            efi_path = alt
        else:
            raise FileNotFoundError("[DIAG] efi_stack.npy not found; run 01_efi_new.py first")
    esi_trend_path = os.path.join(out, "esi_trend.npy")
    if not os.path.exists(esi_trend_path):
        raise FileNotFoundError("[DIAG] esi_trend.npy not found; run 02_esi.py first")
    grid = gio.analysis_grid()
    H, W = grid["height"], grid["width"]
    efi_shape = np.load(efi_path, mmap_mode="r").shape
    if efi_shape[0] < nY:
        raise ValueError(f"[DIAG] efi_stack has {efi_shape[0]} years, need {nY} ({start}-{end})")
    tpath = build_type_map(grid, out)
    if tpath is None:
        print("[DIAG] land-cover file for TYPE_YEAR not found; type breakdown skipped")
    rpath = os.path.join(out, "esi_reliability.tif")
    t = c["TILE"]
    tiles = [(r, cc, min(t, H - r), min(t, W - cc)) for r in range(0, H, t) for cc in range(0, W, t)]
    print(f"[DIAG] grid {H}x{W}{' (ROI)' if c.get('ROI') else ''}, EFI years {start}-{end} (n={nY}) from "
          f"{os.path.basename(efi_path)} {efi_shape}, {len(tiles)} tiles")
    dEFI = np.full((H, W), np.nan, "float32")
    pEFI = np.full((H, W), np.nan, "float32")
    dESI = np.full((H, W), np.nan, "float32")
    reg = np.full((H, W), NOD, "uint8")
    cnt = np.zeros((6, 3, 2, 4), np.int64)
    ucnt = {}
    t0 = time.time()
    gen = Parallel(n_jobs=c["N_JOBS"], return_as="generator_unordered")(
        delayed(process_tile)(tl, efi_path, nY, tpath, rpath, esi_trend_path) for tl in tiles)
    for i, res in enumerate(gen, 1):
        if res is not None:
            r0, c0, h, w, de, pe, ds, rg, ct, uc = res
            dEFI[r0:r0 + h, c0:c0 + w] = de; pEFI[r0:r0 + h, c0:c0 + w] = pe
            dESI[r0:r0 + h, c0:c0 + w] = ds; reg[r0:r0 + h, c0:c0 + w] = rg
            cnt += ct
            for u, a in uc.items():
                ucnt[u] = ucnt.get(u, 0) + a
        if i % 20 == 0 or i == len(tiles):
            el = time.time() - t0
            print(f"[DIAG] tiles {i}/{len(tiles)}  {el:.0f}s  ETA {el / i * (len(tiles) - i):.0f}s", flush=True)
    return dEFI, pEFI, dESI, reg, cnt, ucnt, grid


# ------------------------------------------------------------ demo (synthetic, no data needed)
def generate_demo():
    c = CONFIG
    H = W = 60
    nY = 21
    yy, xx = np.mgrid[0:H, 0:W]
    plateau = (xx >= 2 * W // 3) & (yy >= H // 2)
    north = (yy < H // 2) & (xx < 2 * W // 3)
    efi = np.empty((nY, H, W), "float32")
    esi_tr = np.where(plateau, -0.004, np.where(north, 0.003, 0.0)) + 0.002 * RNG.standard_normal((H, W))
    for t in range(nY):
        efi[t] = np.clip(0.5 + np.where(plateau | north, 0.006, 0.0) * t + 0.01 * RNG.standard_normal((H, W)), 0, 1)
    dE, pE = theil_sen_mk(efi.reshape(nY, -1))
    dEFI = dE.reshape(H, W); pEFI = pE.reshape(H, W)
    reg = classify(dEFI, esi_tr.astype("float32"))
    typ = np.ones((H, W), int); typ[plateau] = 3; typ[north] = 2
    cnt = np.zeros((6, 3, 2, 4), np.int64)
    sig = pEFI < ALPHA
    key = ((typ * 3 + 1) * 2 + sig) * 4 + reg
    cnt += np.bincount(key.ravel(), minlength=144).reshape(6, 3, 2, 4)
    return dEFI, pEFI, esi_tr.astype("float32"), reg, cnt, {}, None


def main():
    c = CONFIG
    out = gio.ensure_outdir()
    have = (os.path.exists(os.path.join(out, "esi_trend.npy")) and
            (os.path.exists(os.path.join(out, "efi_stack.npy")) or
             os.path.exists(os.path.join(gio.PROJECT_ROOT, "output", "efi_stack.npy"))))
    if have:
        dEFI, pEFI, dESI, reg, cnt, ucnt, grid = run_real(out)
    elif c["ALLOW_DEMO"]:
        print("[DIAG] upstream outputs not found; running synthetic demo (run 01_efi_new.py and 02_esi.py first)")
        dEFI, pEFI, dESI, reg, cnt, ucnt, grid = generate_demo()
    else:
        raise FileNotFoundError("[DIAG] need efi_stack.npy and esi_trend.npy")

    np.save(os.path.join(out, "diag_dEFI.npy"), dEFI)
    np.save(os.path.join(out, "diag_dESI.npy"), dESI)
    np.save(os.path.join(out, "diag_regime.npy"), reg)
    if grid is not None:
        gio.write_tif(os.path.join(out, "diag_dEFI.tif"), dEFI, grid)
        gio.write_tif(os.path.join(out, "diag_pEFI.tif"), pEFI, grid)
        gio.write_tif(os.path.join(out, "diag_dESI.tif"), dESI, grid)
        gio.write_tif(os.path.join(out, "diag_regime.tif"), reg, grid, "uint8", NOD)
        print("[DIAG] wrote GeoTIFFs: diag_dEFI / diag_pEFI / diag_dESI / diag_regime")

    rows = shares_rows(cnt, ucnt)
    with open(os.path.join(out, "diag_regime_shares.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "scheme", "n_pixels", *REGIME_NAMES])
        w.writerows(rows)
    for r in rows:
        if r[0] == "ALL" or r[0].startswith("type_"):
            print(f"[DIAG] {r[0]:<22s} {r[1]:<20s} n={r[2]:>12,}  " +
                  "  ".join(f"{n}={v}%" for n, v in zip(REGIME_NAMES, r[3:])))
    allrow = next(r for r in rows if r[0] == "ALL" and r[1] == "all")
    print(f"[DIAG] >>> FRAGILE GREENING = {allrow[4]}%  (headline diagnostic) <<<")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#2e7d32", "#e07b39", "#6b8fb5", "#8e2f2f"])
        sd = max(1, reg.shape[0] // 900)
        g = (reg != NOD)
        ii = np.flatnonzero(g.ravel())
        ii = RNG.choice(ii, min(ii.size, 200_000), replace=False) if ii.size else ii
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
        ax[0].axhline(0, color="k", lw=.6); ax[0].axvline(0, color="k", lw=.6)
        ax[0].scatter(dEFI.ravel()[ii], dESI.ravel()[ii], c=reg.ravel()[ii], cmap=cmap, vmin=0, vmax=3, s=2, alpha=.4)
        ax[0].set_xlabel("dEFI (per year)"); ax[0].set_ylabel("dESI (per window step)")
        ax[0].set_title("Function-stability space")
        rg = np.where(reg == NOD, np.nan, reg).astype("float32")
        im = ax[1].imshow(rg[::sd, ::sd], cmap=cmap, vmin=0, vmax=3); ax[1].set_title("Regime map"); ax[1].axis("off")
        cb = plt.colorbar(im, ax=ax[1], ticks=[0, 1, 2, 3], shrink=.8); cb.ax.set_yticklabels(REGIME_NAMES, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(out, "08_diagnostic_space.png"), dpi=130)
        print(f"[DIAG] figure -> {os.path.join(out, '08_diagnostic_space.png')}")
    except Exception as e:
        print(f"[DIAG] (figure skipped: {e})")
    print("[DIAG] done.")


if __name__ == "__main__":
    main()
