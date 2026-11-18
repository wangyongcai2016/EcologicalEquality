#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 02 — Ecological Stability Index (ESI): resistance & recovery
      [monthly-NPP GeoTIFF | whole-period + moving-window | tiled, parallel]
================================================================================
Paper sections 2.2.2 & 2.7 (deepened version), Eqs. 3-4.

    NPP(t) = Trend(t) + Season(t) + Residual(t)                       (Eq. 3)
    resistance : |slope| of the standardised NPP anomaly on the precipitation anomaly
                 (larger = weaker resistance)
    recovery   : lag-1 temporal autocorrelation TAC (higher = slower recovery)
    ESI = 0.5*[1 - norm(resistance)] + 0.5*[1 - norm(TAC)]            (Eq. 4)

TWO OUTPUTS
  (A) WHOLE-PERIOD ESI : one map over the full record.
  (B) MOVING-WINDOW ESI: per-window ESI series + its Theil-Sen trend (dESI), which
      08_diagnostic_space.py uses.

Normalisation: min-max bounds (1st / 99th percentile) are fixed ONCE and shared by all
windows, so window ESI values are comparable and dESI measures a real change.

Reliability (Smith & Boers 2023): TAC-based resilience estimates are less reliable where
biomass is high. Pixels are split into terciles of mean NPP:
    esi_reliability = 2 lowest-NPP tercile (most reliable) | 1 middle | 0 highest-NPP tercile (low confidence)
TAC is cross-validated against DFA.

--------------------------------------------------------------------------------
HOW TO RUN  (paths come from geo_io.CONFIG; no command-line arguments)
--------------------------------------------------------------------------------
  1. python 01_efi_new.py        (optional but recommended: writes STABLE_TIF = analysis mask)
  2. python 00_align_monthly.py  (once: puts monthly NPP / precipitation / PET on the UNITS_TIF grid)
  3. python 02_esi.py
  Pilot first: set CONFIG["ROI"] = (7050, 7350, 6300, 6600) in geo_io.py, run 00 and 02, check the
  timing printed below, then set ROI back to None for the full run.

INPUT  : aligned monthly NPP + precipitation (NPP_ALIGN_DIR, PRECIP_ALIGN_DIR), UNITS_TIF, STABLE_TIF
OUTPUT (CONFIG['OUT_DIR']):
  Whole-period : esi.tif/.npy, esi_resistance.tif, esi_tac.tif, esi_dfa_alpha.tif, esi_reliability.tif
  Moving-window: esi_window_YYYY.tif (window centre year), esi_stack.npy (n_windows,H,W),
                 esi_trend.tif/.npy (dESI, per window step)
  Preview      : 02_esi_preview.png
================================================================================
"""
from __future__ import annotations
import os, shutil, time, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(7)
MIN_VALID_MONTHS = 24

try:
    from statsmodels.tsa.seasonal import STL
    _HAS_STL = True
except Exception:
    _HAS_STL = False


# ------------------------------------------------------------ STL (Eq. 3)
def stl_residual(series, period=12, robust=True):
    s = np.asarray(series, float)
    idx = np.arange(s.size); good = np.isfinite(s)
    if good.sum() < MIN_VALID_MONTHS:
        return np.full_like(s, np.nan)
    s = s.copy(); s[~good] = np.interp(idx[~good], idx[good], s[good])
    if _HAS_STL:
        return STL(s, period=period, robust=robust).fit().resid
    w = period + (period % 2 == 0)
    trend = np.convolve(np.pad(s, w // 2, mode="edge"), np.ones(w) / w, "valid")
    detr = s - trend; seas = np.zeros_like(s)
    for p in range(period):
        seas[p::period] = np.nanmean(detr[p::period])
    return s - trend - (seas - seas.mean())


# ------------------------------------------------- vectorised metrics, columns = pixels
def corr_abs(A, C):
    """|Pearson r| per column. |r| equals |slope of the standardised regression| (resistance)."""
    a = A - A.mean(0); c = C - C.mean(0)
    den = np.sqrt((a * a).sum(0) * (c * c).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs((a * c).sum(0) / den)
    r[den < 1e-12] = np.nan
    return r


def lag1_autocorr(A):
    a = A - A.mean(0); den = (a * a).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a[:-1] * a[1:]).sum(0) / den
    r[den < 1e-12] = np.nan
    return r


def dfa_alpha(A, scales=(4, 8, 16, 32)):
    T = A.shape[0]
    if T < 2 * max(scales):
        return np.full(A.shape[1], np.nan)
    y = np.cumsum(A - A.mean(0), axis=0); logF = []
    for s in scales:
        n = T // s
        seg = y[:n * s].reshape(n, s, -1)
        tc = np.arange(s) - (s - 1) / 2.0
        slope = (tc[None, :, None] * seg).sum(1) / (tc ** 2).sum()
        fit = seg.mean(1)[:, None, :] + slope[:, None, :] * tc[None, :, None]
        rms = np.sqrt(((seg - fit) ** 2).mean(1))
        logF.append(np.log(np.maximum(rms.mean(0), 1e-12)))
    logF = np.stack(logF); xs = np.log(np.asarray(scales, float)); xc = xs - xs.mean()
    return (xc[:, None] * (logF - logF.mean(0))).sum(0) / (xc ** 2).sum()


def window_starts(T, period, window_years, step):
    return list(range(0, T - window_years * period + 1, step * period))


# ------------------------------------------------- per-pixel core (columns = pixels)
def compute_pixels(npp, clim, period, window_years, step=1, robust=True):
    """npp, clim : (T, n). Returns per-pixel whole-period and per-window raw components."""
    T, n = npp.shape
    ok = ((np.isfinite(npp).sum(0) >= MIN_VALID_MONTHS) &
          (np.isfinite(clim).sum(0) >= MIN_VALID_MONTHS))
    starts = window_starts(T, period, window_years, step)
    out = dict(ok=ok, dens=np.full(n, np.nan, "float32"),
               resist=np.full(n, np.nan, "float32"), tac=np.full(n, np.nan, "float32"),
               dfa=np.full(n, np.nan, "float32"),
               wres=np.full((len(starts), n), np.nan, "float32"),
               wtac=np.full((len(starts), n), np.nan, "float32"))
    cols = np.flatnonzero(ok)
    if cols.size == 0:
        return out
    A = np.empty((T, cols.size)); C = np.empty((T, cols.size))
    for k, j in enumerate(cols):
        A[:, k] = stl_residual(npp[:, j], period, robust)
        C[:, k] = stl_residual(clim[:, j], period, robust)
    out["dens"][cols] = np.nanmean(npp[:, cols], axis=0)
    out["resist"][cols] = corr_abs(A, C)
    out["tac"][cols] = lag1_autocorr(A)
    out["dfa"][cols] = dfa_alpha(A)
    win_m = window_years * period
    for w, s0 in enumerate(starts):
        sl = slice(s0, s0 + win_m)
        out["wres"][w, cols] = corr_abs(A[sl], C[sl])
        out["wtac"][w, cols] = lag1_autocorr(A[sl])
    return out


# ------------------------------------------------- storage (memmap on disk, ndarray for the demo)
def _alloc(store_dir, name, shape, fill_nan):
    if store_dir is None:
        return np.full(shape, np.nan, "float32") if fill_nan else np.zeros(shape, "float32")
    a = np.lib.format.open_memmap(os.path.join(store_dir, name + ".npy"), mode="w+",
                                  dtype="float32", shape=shape)
    if fill_nan:
        a[:] = np.nan
    return a


def _store_paths(store_dir):
    return {k: os.path.join(store_dir, k + ".npy") for k in ("dens", "resist", "tac", "dfa", "wres", "wtac")}


def _write_tile(store, r0, c0, h, w, midx, res):
    ok = res["ok"]; pos = midx[ok]
    for key in ("dens", "resist", "tac", "dfa"):
        full = np.full(h * w, np.nan, "float32"); full[pos] = res[key][ok]
        store[key][r0:r0 + h, c0:c0 + w] = full.reshape(h, w)
    for key in ("wres", "wtac"):
        nW = res[key].shape[0]
        full = np.full((nW, h * w), np.nan, "float32"); full[:, pos] = res[key][:, ok]
        store[key][:, r0:r0 + h, c0:c0 + w] = full.reshape(nW, h, w)


def process_tile(win_t, paths, start, end, period, window_years, robust):
    """One tile: read aligned NPP + precipitation, compute components, write into the memmaps."""
    from rasterio.windows import Window
    r0, c0, h, w = win_t
    win = Window(c0, r0, w, h)
    mask = gio.read_mask(win)
    if not mask.any():
        return 0
    midx = np.flatnonzero(mask.ravel())
    npp, _ = gio.read_monthly_window("npp", start, end, win)
    clim, _ = gio.read_monthly_window("pre", start, end, win)
    npp = npp.reshape(npp.shape[0], -1)[:, midx]; clim = clim.reshape(clim.shape[0], -1)[:, midx]
    res = compute_pixels(npp, clim, period, window_years, 1, robust)
    store = {k: np.load(p, mmap_mode="r+") for k, p in paths.items()}
    _write_tile(store, r0, c0, h, w, midx, res)
    for a in store.values():
        a.flush()
    return int(res["ok"].sum())


# ------------------------------------------------- pass 2: shared normalisation, ESI, trend
def _bounds(x):
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [1, 99]) if x.size else (0.0, 1.0)
    return float(lo), float(hi)


def _norm01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)


def _theil_sen_cols(E):
    """E : (nW, n) -> Theil-Sen slope per column (NaN-aware; needs >= 3 finite windows)."""
    nW, n = E.shape
    ii, jj = np.triu_indices(nW, 1)
    S = (E[jj] - E[ii]) / (jj - ii)[:, None]
    out = np.full(n, np.nan, "float32")
    full = ~np.isnan(S).any(0)
    if full.any():
        out[full] = np.median(S[:, full], axis=0)
    part = ~full & (np.isfinite(E).sum(0) >= 3)
    if part.any():
        out[part] = np.nanmedian(S[:, part], axis=0)
    return out


def finalize(store, out_dir, centres, shape, sample=500_000):
    """Bounds -> ESI (whole-period + windows) -> dESI -> reliability. Returns dict of arrays/paths."""
    H, W = shape; nW = len(centres)
    valid_flat = np.flatnonzero(np.isfinite(store["dens"]).ravel())
    if valid_flat.size == 0:
        raise RuntimeError("[ESI] no valid pixels; check the mask / aligned data")
    take = np.sort(RNG.choice(valid_flat, size=min(sample, valid_flat.size), replace=False))
    lo_r, hi_r = _bounds(store["resist"].ravel()[take])
    lo_t, hi_t = _bounds(store["tac"].ravel()[take])
    wlo_r, whi_r = _bounds(store["wres"].reshape(nW, -1)[:, take].ravel())
    wlo_t, whi_t = _bounds(store["wtac"].reshape(nW, -1)[:, take].ravel())
    print(f"[ESI] shared bounds  whole-period resist [{lo_r:.3f},{hi_r:.3f}] tac [{lo_t:.3f},{hi_t:.3f}] | "
          f"windows resist [{wlo_r:.3f},{whi_r:.3f}] tac [{wlo_t:.3f},{whi_t:.3f}]")

    esi = 0.5 * (1 - _norm01(store["resist"], lo_r, hi_r)) + 0.5 * (1 - _norm01(store["tac"], lo_t, hi_t))
    esi = esi.astype("float32")

    if out_dir is None:
        stack = np.full((nW, H, W), np.nan, "float32")
    else:
        stack = np.lib.format.open_memmap(os.path.join(out_dir, "esi_stack.npy"), mode="w+",
                                          dtype="float32", shape=(nW, H, W))
    trend = np.full((H, W), np.nan, "float32")
    step_rows = 32
    for r0 in range(0, H, step_rows):
        r1 = min(H, r0 + step_rows)
        wr = np.asarray(store["wres"][:, r0:r1]); wt = np.asarray(store["wtac"][:, r0:r1])
        E = 0.5 * (1 - _norm01(wr, wlo_r, whi_r)) + 0.5 * (1 - _norm01(wt, wlo_t, whi_t))
        E = E.astype("float32")
        E[:, ~np.isfinite(store["dens"][r0:r1])] = np.nan
        stack[:, r0:r1] = E
        v = np.isfinite(E).any(0)
        if v.any():
            t = np.full(v.shape, np.nan, "float32")
            t[v] = _theil_sen_cols(E[:, v])
            trend[r0:r1] = t
    if out_dir is not None:
        stack.flush()

    dens = np.asarray(store["dens"]); d = dens[np.isfinite(dens)]
    q33, q66 = np.percentile(d, [100 / 3, 200 / 3])
    reliab = np.full((H, W), np.nan, "float32")
    reliab[np.isfinite(dens) & (dens > q66)] = 0      # highest-NPP tercile: low confidence
    reliab[np.isfinite(dens) & (dens > q33) & (dens <= q66)] = 1
    reliab[np.isfinite(dens) & (dens <= q33)] = 2     # lowest-NPP tercile: most reliable
    return dict(esi=esi, stack=stack, trend=trend, reliab=reliab)


# ------------------------------------------------------------ demo (synthetic, no data needed)
def generate_demo():
    c = CONFIG
    H = W = 40; n_years = c["END_YEAR"] - c["START_YEAR"] + 1; T = n_years * 12
    yy, xx = np.mgrid[0:H, 0:W]; grass = np.ones((H, W), bool)
    grass[(xx - 2) ** 2 + (yy - 2) ** 2 < 4] = False
    plateau = (xx >= 2 * W // 3) & (yy >= H // 2)
    season = 1 + 0.6 * np.sin(2 * np.pi * (np.arange(T) % 12) / 12)
    npp = np.empty((T, H, W)); clim = np.empty((T, H, W)); base = 40 + 60 * RNG.random((H, W))
    for m in range(T):
        yr = m // 12; vscale = np.where(plateau, 1 + 0.06 * yr, 1.0)
        npp[m] = base * season[m] + vscale * 6 * RNG.standard_normal((H, W))
        clim[m] = 50 * season[m] + 10 * RNG.standard_normal((H, W))
    npp[:, ~grass] = np.nan; clim[:, ~grass] = np.nan
    return npp, clim, grass


# ------------------------------------------------------------ main
def _aligned_ready():
    return all(gio.files_available(gio.align_dir(k)) for k in ("NPP_ALIGN_DIR", "PRECIP_ALIGN_DIR"))


def run_real(out):
    from joblib import Parallel, delayed
    c = CONFIG; period = c["STL_PERIOD"]; win_y = c["ESI_WINDOW"]
    start, end = c["START_YEAR"], c["END_YEAR"]
    grid = gio.analysis_grid(); H, W = grid["height"], grid["width"]
    T = (end - start + 1) * 12
    starts = window_starts(T, period, win_y, 1)
    centres = np.array([start + s // period + win_y // 2 for s in starts])
    nW = len(centres)
    tmp = os.path.join(out, "_esi_tmp"); os.makedirs(tmp, exist_ok=True)
    print(f"[ESI] grid {H}x{W}{' (ROI)' if c.get('ROI') else ''}, months {T}, {nW} windows "
          f"{centres[0]}-{centres[-1]}, robust STL={c['STL_ROBUST']}, workers={c['N_JOBS']}")

    for k, shp in (("dens", (H, W)), ("resist", (H, W)), ("tac", (H, W)), ("dfa", (H, W))):
        _alloc(tmp, k, shp, True)
    for k in ("wres", "wtac"):
        _alloc(tmp, k, (nW, H, W), False)       # only valid pixels are ever read back
    paths = _store_paths(tmp)

    t = c["TILE"]
    tiles = [(r, cc, min(t, H - r), min(t, W - cc)) for r in range(0, H, t) for cc in range(0, W, t)]
    t0 = time.time(); npx = 0
    gen = Parallel(n_jobs=c["N_JOBS"], return_as="generator_unordered")(
        delayed(process_tile)(tl, paths, start, end, period, win_y, c["STL_ROBUST"]) for tl in tiles)
    for i, n in enumerate(gen, 1):
        npx += n
        if i % 10 == 0 or i == len(tiles):
            el = time.time() - t0
            print(f"[ESI] tiles {i}/{len(tiles)}  pixels {npx:,}  {el:.0f}s  "
                  f"ETA {el / i * (len(tiles) - i):.0f}s", flush=True)
    if npx:
        print(f"[ESI] throughput {npx / max(time.time() - t0, 1e-9):.0f} px/s "
              f"({(time.time() - t0) / npx * 1000:.2f} ms/px wall)")
    store = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    res = finalize(store, out, centres, (H, W))
    return store, res, centres, grid, tmp


def run_demo(out):
    c = CONFIG; period = c["STL_PERIOD"]; win_y = c["ESI_WINDOW"]
    npp, clim, grass = generate_demo()
    T, H, W = npp.shape
    starts = window_starts(T, period, win_y, 1)
    centres = np.array([c["START_YEAR"] + s // period + win_y // 2 for s in starts])
    nW = len(centres)
    store = {k: _alloc(None, k, (H, W), True) for k in ("dens", "resist", "tac", "dfa")}
    store.update({k: _alloc(None, k, (nW, H, W), True) for k in ("wres", "wtac")})
    midx = np.flatnonzero(grass.ravel())
    res = compute_pixels(npp.reshape(T, -1)[:, midx], clim.reshape(T, -1)[:, midx], period, win_y,
                         1, c["STL_ROBUST"])
    _write_tile(store, 0, 0, H, W, midx, res)
    fin = finalize(store, None, centres, (H, W))
    np.save(os.path.join(out, "esi_stack.npy"), fin["stack"])
    return store, fin, centres, None, None


def main():
    c = CONFIG; out = gio.ensure_outdir()
    real_cfg = gio.files_available(c["UNITS_TIF"]) and gio.files_available(c["NPP_DIR"])
    if real_cfg and not _aligned_ready():
        raise FileNotFoundError(
            "[ESI] aligned monthly rasters not found in "
            f"{gio.align_dir('NPP_ALIGN_DIR')} / {gio.align_dir('PRECIP_ALIGN_DIR')}.\n"
            "      Run 00_align_monthly.py first (the raw NPP / precipitation / PET grids differ from UNITS_TIF).")
    if real_cfg:
        store, res, centres, grid, tmp = run_real(out)
    elif c["ALLOW_DEMO"]:
        print("[ESI] data paths not found; running synthetic demo (edit paths in geo_io.py CONFIG)")
        store, res, centres, grid, tmp = run_demo(out)
    else:
        raise FileNotFoundError("[ESI] configured data not found and ALLOW_DEMO=False")

    esi, trend, reliab, stack = res["esi"], res["trend"], res["reliab"], res["stack"]
    valid = np.isfinite(store["dens"])
    np.save(os.path.join(out, "esi.npy"), esi)
    np.save(os.path.join(out, "esi_trend.npy"), trend)
    if grid is not None:
        for arr, fn in [(esi, "esi"), (store["resist"], "esi_resistance"), (store["tac"], "esi_tac"),
                        (store["dfa"], "esi_dfa_alpha"), (reliab, "esi_reliability"), (trend, "esi_trend")]:
            gio.write_tif(os.path.join(out, fn + ".tif"), np.asarray(arr, "float32"), grid)
        for w, yr in enumerate(centres):
            gio.write_tif(os.path.join(out, f"esi_window_{yr}.tif"), np.asarray(stack[w], "float32"), grid)

    g = valid & np.isfinite(store["tac"]) & np.isfinite(store["dfa"])
    if g.sum() > 5:
        idx = np.flatnonzero(g.ravel()); idx = RNG.choice(idx, min(idx.size, 2_000_000), replace=False)
        r = np.corrcoef(np.asarray(store["tac"]).ravel()[idx], np.asarray(store["dfa"]).ravel()[idx])[0, 1]
        print(f"[ESI] TAC vs DFA cross-validation r = {r:.3f}")
    print(f"[ESI] valid pixels {int(valid.sum()):,}; whole-period mean ESI = {np.nanmean(esi):.3f}; "
          f"most-reliable (lowest-NPP tercile) {100 * (reliab == 2).sum() / max(valid.sum(), 1):.1f}%")
    dec = np.nansum(trend[valid] < 0) / max(valid.sum(), 1) * 100
    print(f"[ESI] {len(centres)} windows, centres {centres[0]}-{centres[-1]}; "
          f"ESI declining (dESI<0) on {dec:.1f}% of pixels")
    print(f"[ESI] saved esi_stack.npy {tuple(stack.shape)} (read by 08_diagnostic_space)")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        H = esi.shape[0]; sd = max(1, H // 900)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        im0 = ax[0].imshow(esi[::sd, ::sd], cmap="RdYlGn", vmin=0, vmax=1); ax[0].set_title("ESI (whole period)")
        plt.colorbar(im0, ax=ax[0], shrink=.8)
        lim = np.nanpercentile(trend, 98) if np.isfinite(trend).any() else 0.02
        im1 = ax[1].imshow(trend[::sd, ::sd], cmap="RdBu", vmin=-lim, vmax=lim); ax[1].set_title("dESI (per window step)")
        plt.colorbar(im1, ax=ax[1], shrink=.8)
        im2 = ax[2].imshow(reliab[::sd, ::sd], cmap="viridis", vmin=0, vmax=2); ax[2].set_title("Reliability (2 = high)")
        plt.colorbar(im2, ax=ax[2], shrink=.8)
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(out, "02_esi_preview.png"), dpi=120)
        print(f"[ESI] preview -> {os.path.join(out, '02_esi_preview.png')}")
    except Exception as e:
        print(f"[ESI] (preview skipped: {e})")

    if tmp:
        del store
        shutil.rmtree(tmp, ignore_errors=True)
    print("[ESI] done.")


if __name__ == "__main__":
    main()
