#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 10 — Spatial early-warning signals (test of Hypothesis H3)   [tiled, parallel | national natural ecosystems]
================================================================================
Paper section 3.3: "if fragile greening reflects approach to a degradation threshold, the eroding-stability zone
should display spatial early-warning signals independent of the ESI construction."

For every month of EWS_YEARS the NPP ANOMALY (value minus the pixel's calendar-month mean over EWS_YEARS) is
computed inside each tile (+ a halo, so windows are not cut at tile borders) and two spatial resilience-loss
indicators are evaluated in a WIN x WIN moving window (Scheffer et al. 2009; Dakos et al. 2012):
    spatial variance          and          spatial lag-1 autocorrelation (Moran's-I-like, right/down neighbours)
then averaged over the months. The 94 GB national anomaly stack is never written.

EWS_YEARS = 2011-2024: from 2000 to 2010 the GLASS NPP of ~1.3 M analysed pixels is exactly 0 (02_esi.py), which
would turn the anomalies of those pixels into artefacts. Anomalies are deseasonalised but NOT detrended (the
climatology is a calendar-month mean), so a monotone NPP trend contributes to the variance.

Comparison between regimes (08_diagnostic_space: 0 true recovery, 1 fragile greening, 2 low-but-stable, 3 collapse):
  * ews_by_regime.csv  mean +/- sd of each indicator for all analysed pixels per regime (descriptive).
  * ews_by_type.csv    fragile greening vs true recovery within each ecosystem type (eco_type.tif): one-sided
    Mann-Whitney U on a LATTICE of pixels THIN_STEP apart (pixels are spatially autocorrelated, so a test on all
    ~10^7 pixels would give p ~ 0 for any difference), Cliff's delta (= 2 AUC - 1) and the share of types in which
    fragile greening is higher.

INPUT  (OUT_DIR): diag_regime.npy, eco_type.tif (08); aligned monthly NPP (G:\\NPP\\NPP_Align), UNITS_TIF, STABLE_TIF
OUTPUT (OUT_DIR): ews_spatial_variance/autocorr .npy + .tif, ews_by_regime.csv, ews_by_type.csv, 10_ews_preview.png
Run:  python 10_early_warning.py        (python 10_early_warning.py --demo: synthetic, writes to OUT_DIR/_demo)
      EQI_ROI / EQI_N_JOBS pilot switches as in 02_esi.py.
================================================================================
"""
from __future__ import annotations
import os, sys, csv, time, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(31)
EWS_YEARS = (2011, 2024)
WIN = 5                       # spatial window (pixels, 2.5 km)
TILE = 512
HALO = WIN // 2 + 1
THIN_STEP = 20                # lattice spacing (pixels = 10 km) for the significance tests
REGIMES = {0: "true_recovery", 1: "fragile_greening", 2: "low_but_stable", 3: "collapse"}
TYPES = {1: "Forest", 2: "Shrub", 3: "Grassland", 4: "Wetland", 5: "Sparse"}


def spatial_variance(field, win):
    """Windowed spatial variance via E[x^2]-E[x]^2 (uniform_filter = box mean)."""
    m = np.isfinite(field)
    x = np.where(m, field, 0.0)
    cnt = np.maximum(uniform_filter(m.astype(float), win), 1e-9)
    mean = uniform_filter(x, win) / cnt
    mean2 = uniform_filter(x * x, win) / cnt
    var = mean2 - mean * mean
    var[~m] = np.nan
    return var


def spatial_autocorr(field, win):
    """Windowed lag-1 spatial autocorrelation (mean of the products of standardised anomalies with the right and
    lower neighbour). A local Moran's-I-like statistic."""
    m = np.isfinite(field)
    x = np.where(m, field, 0.0)
    cnt = np.maximum(uniform_filter(m.astype(float), win), 1e-9)
    mean = uniform_filter(x, win) / cnt
    anom = np.where(m, field - mean, 0.0)
    var = uniform_filter(anom * anom, win) / cnt + 1e-12
    prod = np.zeros_like(anom)
    prod[:, :-1] += anom[:, :-1] * anom[:, 1:]
    prod[:-1, :] += anom[:-1, :] * anom[1:, :]
    ac = (uniform_filter(prod, win) / cnt) / var
    ac[~m] = np.nan
    return ac


def ews_stack(anom, grass, win):
    """Time-averaged spatial variance and autocorrelation of an anomaly stack (T, h, w); NaN outside `grass`."""
    v_acc = np.zeros(grass.shape); a_acc = np.zeros(grass.shape); n = np.zeros(grass.shape)
    for t in range(anom.shape[0]):
        f = np.where(grass, anom[t], np.nan)
        ok = np.isfinite(f)
        if not ok.any():
            continue
        v = spatial_variance(f, win); a = spatial_autocorr(f, win)
        g = ok & np.isfinite(v) & np.isfinite(a)
        v_acc[g] += v[g]; a_acc[g] += a[g]; n[g] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(n > 0, v_acc / n, np.nan).astype("float32"), np.where(n > 0, a_acc / n, np.nan).astype("float32"))


def _tile(r0, c0, h, w, H, W, y0, y1):
    from rasterio.windows import Window
    rr0, rr1 = max(r0 - HALO, 0), min(r0 + h + HALO, H)
    cc0, cc1 = max(c0 - HALO, 0), min(c0 + w + HALO, W)
    win = Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
    mask = gio.read_mask(win)
    ir, ic = r0 - rr0, c0 - cc0
    if not mask[ir:ir + h, ic:ic + w].any():
        return None
    npp, _ = gio.read_monthly_window("npp", y0, y1, win)
    T = npp.shape[0]
    clim = np.nanmean(npp.reshape(T // 12, 12, *npp.shape[1:]), axis=0)             # (12, hh, ww)
    anom = npp - np.tile(clim, (T // 12, 1, 1))
    zero = float(((npp == 0) & mask[None]).sum()); tot = float((np.isfinite(npp) & mask[None]).sum())
    var, ac = ews_stack(anom, mask, WIN)
    return r0, c0, var[ir:ir + h, ic:ic + w], ac[ir:ir + h, ic:ic + w], zero, tot


def run_real(out):
    from joblib import Parallel, delayed
    grid = gio.analysis_grid(); H, W = grid["height"], grid["width"]
    y0, y1 = EWS_YEARS
    tiles = [(r, c, min(TILE, H - r), min(TILE, W - c), H, W, y0, y1) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    print(f"[EWS] grid {H}x{W}{' (ROI)' if CONFIG.get('ROI') else ''}, NPP {y0}-{y1} ({(y1 - y0 + 1) * 12} months), "
          f"window {WIN}, {len(tiles)} tiles of {TILE}, {CONFIG['N_JOBS']} workers", flush=True)
    var = np.full((H, W), np.nan, "float32"); ac = var.copy(); zero = tot = 0.0; t0 = time.time()
    gen = Parallel(n_jobs=CONFIG["N_JOBS"], return_as="generator_unordered")(delayed(_tile)(*t) for t in tiles)
    for i, r in enumerate(gen, 1):
        if r is not None:
            r0, c0, v, a, z, n = r
            var[r0:r0 + v.shape[0], c0:c0 + v.shape[1]] = v; ac[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a
            zero += z; tot += n
        if i % 10 == 0 or i == len(tiles):
            el = time.time() - t0
            print(f"[EWS] tiles {i}/{len(tiles)}  {el:.0f}s  ETA {el / i * (len(tiles) - i):.0f}s", flush=True)
    print(f"[EWS] share of exactly-zero NPP among analysed pixel-months in {y0}-{y1}: {100 * zero / max(tot, 1):.2f}%")
    return var, ac, grid


def cliffs(a, b):
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(a, b, alternative="greater")
    return float(2 * u / (a.size * b.size) - 1), float(p)


def compare(var, ac, regime, etype, out, tag=""):
    ok = np.isfinite(var) & np.isfinite(ac)
    rows = []
    for k, nm in REGIMES.items():
        s = ok & (regime == k)
        if s.any():
            rows.append([nm, f"{np.mean(var[s]):.6g}", f"{np.std(var[s]):.6g}", f"{np.mean(ac[s]):.4f}", f"{np.std(ac[s]):.4f}", int(s.sum())])
    with open(os.path.join(out, "ews_by_regime.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["regime", "var_mean", "var_sd", "ac_mean", "ac_sd", "n"]); w.writerows(rows)
    print(f"[EWS]{tag} per-regime spatial variance / autocorrelation (all analysed pixels):")
    for r in rows:
        print(f"        {r[0]:17s} var={r[1]:>10}  ac={r[3]:>7}  (n={r[5]:,})")
    H, W = var.shape
    lat = np.zeros((H, W), bool); lat[THIN_STEP // 2::THIN_STEP, THIN_STEP // 2::THIN_STEP] = True
    trows = []; higher = {"var": 0, "ac": 0}; ntypes = 0
    for t, tn in [(0, "all types")] + list(TYPES.items()):
        sel = ok & lat & ((etype == t) if t else True)
        fg, tr = sel & (regime == 1), sel & (regime == 0)
        if fg.sum() < 30 or tr.sum() < 30:
            trows.append([tn, int(fg.sum()), int(tr.sum()), "", "", "", ""]); continue
        dv, pv = cliffs(var[fg], var[tr]); da, pa = cliffs(ac[fg], ac[tr])
        trows.append([tn, int(fg.sum()), int(tr.sum()), f"{dv:+.3f}", f"{pv:.3g}", f"{da:+.3f}", f"{pa:.3g}"])
        if t:
            ntypes += 1; higher["var"] += dv > 0; higher["ac"] += da > 0
    with open(os.path.join(out, "ews_by_type.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ecosystem_type", "n_fragile_thinned", "n_recovery_thinned", "cliffs_delta_var", "p_var_greater",
                    "cliffs_delta_ac", "p_ac_greater"]); w.writerows(trows)
    print(f"[EWS]{tag} fragile greening vs true recovery on a {THIN_STEP}-pixel lattice (Cliff's delta > 0: fragile higher; "
          f"one-sided Mann-Whitney p):")
    for r in trows:
        print(f"        {r[0]:10s} n={r[1]:>6}/{r[2]:<6} var: delta {r[3]:>7} p {r[4]:>8}   ac: delta {r[5]:>7} p {r[6]:>8}")
    if ntypes:
        print(f"[EWS]{tag} fragile greening higher in {higher['var']}/{ntypes} types (variance), {higher['ac']}/{ntypes} types "
              f"(autocorrelation)")


def generate_demo(H=60, W=60, n_times=48):
    yy, xx = np.mgrid[0:H, 0:W]
    plateau = (xx >= 2 * W // 3) & (yy >= H // 2); north = (yy < H // 2) & (xx < 2 * W // 3)
    regime = np.full((H, W), 3, "uint8"); regime[north] = 0; regime[plateau] = 1
    etype = np.full((H, W), 3, "uint8")
    anom = np.empty((n_times, H, W), "float32")
    for t in range(n_times):
        raw = np.where(plateau, 1 + 0.05 * t, 1.0) * RNG.standard_normal((H, W))
        anom[t] = np.where(plateau, 0.5 * raw + uniform_filter(raw, 3), raw)
    return anom, np.ones((H, W), bool), regime, etype


def main():
    out = gio.ensure_outdir()
    if "--demo" in sys.argv:
        print("[EWS] *** DEMO MODE: synthetic data, outputs go to OUT_DIR/_demo and mean nothing ***")
        out = os.path.join(out, "_demo"); os.makedirs(out, exist_ok=True)
        anom, grass, regime, etype = generate_demo()
        var, ac = ews_stack(anom, grass, WIN)
        compare(var, ac, regime, etype, out, " DEMO")
        return
    rp = os.path.join(out, "diag_regime.npy"); tp = os.path.join(out, "eco_type.tif")
    miss = [p for p in (rp, tp) if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError(f"[EWS] missing {miss}: run 08_diagnostic_space.py first (10 has no silent synthetic fallback; "
                                f"--demo for a code test)")
    var, ac, grid = run_real(out)
    np.save(os.path.join(out, "ews_spatial_variance.npy"), var); np.save(os.path.join(out, "ews_spatial_autocorr.npy"), ac)
    if gio._HAS_RIO:
        gio.write_tif(os.path.join(out, "ews_spatial_variance.tif"), var, grid)
        gio.write_tif(os.path.join(out, "ews_spatial_autocorr.tif"), ac, grid)
    regime = np.load(rp, mmap_mode="r"); etype = gio.read_tif(tp)[0]
    roi = CONFIG.get("ROI")
    if roi:
        regime = regime[roi[0]:roi[1], roi[2]:roi[3]]; etype = etype[roi[0]:roi[1], roi[2]:roi[3]]
    regime = np.asarray(regime); etype = np.nan_to_num(etype, nan=0).astype("uint8")
    compare(var, ac, regime, etype, out)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        k = max(1, max(var.shape) // 800)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        a0 = ax[0].imshow(var[::k, ::k], cmap="magma", vmax=np.nanpercentile(var, 99)); ax[0].set_title("Spatial variance of NPP anomalies"); plt.colorbar(a0, ax=ax[0], shrink=.8)
        a1 = ax[1].imshow(ac[::k, ::k], cmap="cividis", vmin=np.nanpercentile(ac, 1), vmax=np.nanpercentile(ac, 99)); ax[1].set_title("Spatial autocorrelation"); plt.colorbar(a1, ax=ax[1], shrink=.8)
        plt.tight_layout(); plt.savefig(os.path.join(out, "10_ews_preview.png"), dpi=120)
        print(f"[EWS] preview -> {out}/10_ews_preview.png")
    except Exception as e:
        print(f"[EWS] (preview skipped: {e})")
    print("[EWS] done.")


if __name__ == "__main__":
    main()
