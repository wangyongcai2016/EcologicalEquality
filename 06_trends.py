#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 06 — Trend, significance & persistence   [no-args / VSCode-run]
================================================================================
Paper 2.3: Theil-Sen slope + Mann-Kendall + Hurst + persistence class, per pixel
on the ANNUAL EQI stack written by 05_eqi.py (eqi_stack.npy).

06 only READS eqi_stack.npy; it no longer rebuilds it. The calendar years come from
eqi_yearly_mean.csv (geo_io.eqi_annual_years), so the stack may start in 2000 or
2002 without any year being mislabelled.

CONFIG["TREND_YEARS"]:
    "core" (default) the years in which ESI and EPI are both observed (normally 2002-2020);
    "all"            every year of the stack (2000-2024; ESI/EPI extrapolated at the ends).
With "core" the Theil-Sen slope equals 05's eqi_trend_core.npy (checked at the end).

Pixels whose series is complete (all selected years finite) are processed vectorised;
pixels with gaps fall back to the scalar functions. Row blocks run in parallel (N_JOBS).

INPUT (CONFIG['OUT_DIR']): eqi_stack.npy, eqi_yearly_mean.csv
OUTPUT: trend_slope/pvalue/code/hurst/persist .npy (+ .tif), trend_years.csv, 06_trend_preview.png
================================================================================
"""
from __future__ import annotations
import os, csv, time, warnings
import numpy as np
from scipy.stats import norm
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(9)
ALPHA = 0.05
ROWS = 32                      # rows per parallel block


# ----------------------------------------------------------------- scalar versions (gap fallback, reference)
def theil_sen(y, x=None):
    """Sen slope per time step. x = time of each value (default 0..T-1); gaps keep their real spacing."""
    y = np.asarray(y, float); x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, float)
    m = np.isfinite(y); y, x = y[m], x[m]; n = y.size
    if n < 3: return np.nan
    sl = [(y[i+1:]-y[i])/(x[i+1:]-x[i]) for i in range(n-1)]
    return float(np.median(np.concatenate(sl)))


def mann_kendall(y, alpha=ALPHA):
    y = np.asarray(y, float); y = y[np.isfinite(y)]; n = y.size
    if n < 3: return np.nan, 0
    s = sum(np.sign(y[k+1:] - y[k]).sum() for k in range(n-1))
    uniq, cnt = np.unique(y, return_counts=True)
    var = (n*(n-1)*(2*n+5) - np.sum(cnt*(cnt-1)*(2*cnt+5))) / 18.0
    z = 0.0 if var <= 0 else ((s-1)/np.sqrt(var) if s > 0 else (s+1)/np.sqrt(var) if s < 0 else 0.0)
    p = 2*(1 - norm.cdf(abs(z)))
    code = 1 if (p < alpha and z > 0) else -1 if (p < alpha and z < 0) else 0
    return float(p), code


def hurst(y, min_chunk=4):
    x = np.asarray(y, float); x = x[np.isfinite(x)]; n = x.size
    if n < min_chunk*2: return np.nan
    sizes = np.unique(np.floor(np.logspace(np.log10(min_chunk), np.log10(n//2), 8)).astype(int))
    rs, used = [], []
    for w in sizes:
        k = n // w
        if k < 1: continue
        vals = []
        for cc in range(k):
            seg = x[cc*w:(cc+1)*w]; dev = np.cumsum(seg - seg.mean())
            R = dev.max() - dev.min(); S = seg.std()
            if S > 0: vals.append(R/S)
        if vals: rs.append(np.mean(vals)); used.append(w)
    return float(np.polyfit(np.log(used), np.log(rs), 1)[0]) if len(used) >= 2 else np.nan


def persistence(slope, h):
    if not (np.isfinite(slope) and np.isfinite(h)): return np.nan
    persist = h > 0.5
    if slope > 0: return 2 if persist else 1
    if slope < 0: return -2 if persist else -1
    return 0


# ----------------------------------------------------------------- vectorised versions, Y = (T, n) complete columns
def theil_sen_vec(Y):
    T = Y.shape[0]; ii, jj = np.triu_indices(T, 1)
    S = (Y[jj] - Y[ii]) / (jj - ii).astype(Y.dtype)[:, None]
    return np.median(S, axis=0)


def mann_kendall_vec(Y, alpha=ALPHA):
    T, n = Y.shape; ii, jj = np.triu_indices(T, 1)
    s = np.sign(Y[jj] - Y[ii]).sum(0).astype("float64")
    tie = np.zeros(n)
    srt = np.sort(Y, axis=0)
    has_tie = (srt[1:] == srt[:-1]).any(0)
    for c in np.flatnonzero(has_tie):                       # ties are rare: handle those columns exactly
        _, cnt = np.unique(Y[:, c], return_counts=True)
        tie[c] = np.sum(cnt * (cnt - 1) * (2 * cnt + 5))
    var = (T * (T - 1) * (2 * T + 5) - tie) / 18.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.sqrt(var)
        z = np.where(var <= 0, 0.0, np.where(s > 0, (s - 1) / sd, np.where(s < 0, (s + 1) / sd, 0.0)))
    p = 2 * (1 - norm.cdf(np.abs(z)))
    code = np.where((p < alpha) & (z > 0), 1, np.where((p < alpha) & (z < 0), -1, 0))
    return p, code


def hurst_vec(Y, min_chunk=4):
    T, n = Y.shape
    if T < min_chunk * 2:
        return np.full(n, np.nan)
    sizes = np.unique(np.floor(np.logspace(np.log10(min_chunk), np.log10(T // 2), 8)).astype(int))
    X = Y.astype("float64"); lw, lr = [], []
    for w in sizes:
        k = T // w
        if k < 1: continue
        seg = X[:k * w].reshape(k, w, n)
        dev = np.cumsum(seg - seg.mean(1, keepdims=True), axis=1)
        R = dev.max(1) - dev.min(1); S = seg.std(1)
        good = S > 0
        ratio = np.where(good, R / np.where(good, S, 1.0), np.nan)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(ratio, axis=0)                       # mean over chunks with S > 0
        lw.append(np.log(w)); lr.append(np.log(m))
    if len(lw) < 2:
        return np.full(n, np.nan)
    lw = np.array(lw)[:, None] * np.ones((1, n)); lr = np.array(lr)
    ok = np.isfinite(lr); cnt = ok.sum(0)
    xw = np.where(ok, lw, 0.0); yr = np.where(ok, lr, 0.0)
    mx = xw.sum(0) / np.maximum(cnt, 1); my = yr.sum(0) / np.maximum(cnt, 1)
    sxx = (np.where(ok, lw - mx, 0.0) ** 2).sum(0); sxy = (np.where(ok, (lw - mx) * (lr - my), 0.0)).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = sxy / sxx
    h[(cnt < 2) | (sxx <= 0)] = np.nan
    return h


def persistence_vec(slope, h):
    out = np.full(slope.shape, np.nan)
    ok = np.isfinite(slope) & np.isfinite(h)
    persist = h > 0.5
    out[ok & (slope > 0)] = np.where(persist, 2, 1)[ok & (slope > 0)]
    out[ok & (slope < 0)] = np.where(persist, -2, -1)[ok & (slope < 0)]
    out[ok & (slope == 0)] = 0
    return out


def trend_cols(Y):
    """Y : (T, n) float. Returns dict of length-n arrays; complete columns vectorised, others scalar."""
    T, n = Y.shape
    res = {k: np.full(n, np.nan, "float64") for k in ("slope", "pval", "code", "hurst", "persist")}
    fin = np.isfinite(Y)
    full = fin.all(0)
    if full.any():
        Yf = Y[:, full].astype("float64")
        sl = theil_sen_vec(Yf); p, cd = mann_kendall_vec(Yf); h = hurst_vec(Yf)
        res["slope"][full] = sl; res["pval"][full] = p; res["code"][full] = cd; res["hurst"][full] = h
        res["persist"][full] = persistence_vec(sl, h)
    part = np.flatnonzero(~full & (fin.sum(0) >= 3))
    for c in part:
        s = Y[:, c]
        res["slope"][c] = theil_sen(s, np.arange(T)); res["pval"][c], res["code"][c] = mann_kendall(s)
        res["hurst"][c] = hurst(s); res["persist"][c] = persistence(res["slope"][c], res["hurst"][c])
    return res


def _block(path, sel, r0, r1):
    """Worker: trends for rows r0:r1 of the selected years."""
    st = np.load(path, mmap_mode="r")
    Y = np.asarray(st[sel, r0:r1], "float32")                 # (T, rows, W)
    T, h, w = Y.shape
    Yc = Y.reshape(T, h * w)
    use = np.isfinite(Yc).sum(0) >= 3
    out = {k: np.full(h * w, np.nan, "float32") for k in ("slope", "pval", "code", "hurst", "persist")}
    if use.any():
        r = trend_cols(Yc[:, use])
        for k in out:
            out[k][use] = r[k]
    return r0, {k: v.reshape(h, w) for k, v in out.items()}


def run(path, sel, rows=ROWS, n_jobs=None):
    st = np.load(path, mmap_mode="r"); H, W = st.shape[1:]
    res = {k: np.full((H, W), np.nan, "float32") for k in ("slope", "pval", "code", "hurst", "persist")}
    starts = list(range(0, H, rows)); nj = min(n_jobs or CONFIG["N_JOBS"], len(starts))
    t0 = time.time()
    if nj > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=nj, verbose=0)(delayed(_block)(path, sel, r0, min(H, r0 + rows)) for r0 in starts)
    else:
        parts = [_block(path, sel, r0, min(H, r0 + rows)) for r0 in starts]
    for r0, d in parts:
        for k in res:
            res[k][r0:r0 + d[k].shape[0]] = d[k]
    print(f"[TREND] {len(starts)} blocks, {nj} workers, {time.time() - t0:.0f}s")
    return res


def generate_demo():
    H = W = 40; n = 19
    yy, xx = np.mgrid[0:H, 0:W]
    plateau = (xx >= 2*W//3) & (yy >= H//2); stack = np.empty((n, H, W), "float32")
    base = 0.5 + 0.1*RNG.standard_normal((H, W)); trend = np.where(plateau, -0.004, 0.006)
    for t in range(n):
        stack[t] = np.clip(base + trend*t + 0.02*RNG.standard_normal((H, W)), 0, 1)
    return stack


def main():
    c = CONFIG; out = gio.ensure_outdir()
    path = os.path.join(out, "eqi_stack.npy")
    if os.path.exists(path):
        years, core = gio.eqi_annual_years(out)
    elif c["ALLOW_DEMO"]:
        print("[TREND] eqi_stack.npy not found (run 05_eqi.py first); running synthetic demo")
        path = os.path.join(out, "_demo_eqi_stack.npy"); np.save(path, generate_demo())
        years = np.arange(2002, 2021); core = np.ones(19, bool)
    else:
        raise FileNotFoundError("[TREND] need eqi_stack.npy from 05_eqi.py")
    mode = str(c.get("TREND_YEARS", "core")).lower()
    sel = np.flatnonzero(core) if mode == "core" else np.arange(len(years))
    if len(sel) < 8:
        print(f"[TREND] WARNING: only {len(sel)} years selected; Hurst needs >= 8")
    print(f"[TREND] annual EQI stack {path}: {len(years)} yr ({years[0]}-{years[-1]}); "
          f"using TREND_YEARS='{mode}': {years[sel][0]}-{years[sel][-1]} ({len(sel)} yr)")
    with open(os.path.join(out, "trend_years.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["year", "used"])
        wr.writerows([[int(y), int(i in set(sel.tolist()))] for i, y in enumerate(years)])

    prof = None
    tif = os.path.join(out, "efi_mean.tif")
    if gio._HAS_RIO and os.path.exists(tif):
        import rasterio
        with rasterio.open(tif) as s:
            prof = s.profile

    res = run(path, sel)
    for k, fn in [("slope","trend_slope"),("pval","trend_pvalue"),("code","trend_code"),
                  ("hurst","trend_hurst"),("persist","trend_persist")]:
        np.save(os.path.join(out, fn+".npy"), res[k])
        if prof is not None:
            gio.write_tif(os.path.join(out, fn+".tif"), res[k], prof)

    sl = res["slope"]; v = np.isfinite(sl); nv = int(v.sum())
    if nv:
        inc = (sl[v] > 0).mean()*100; dec = (sl[v] < 0).mean()*100
        hv = np.isfinite(res["hurst"]); hp = (res["hurst"][hv] > 0.5).mean()*100
        pdec = (res["persist"][v] == -2).mean()*100
        sig_up = (res["code"][v] == 1).mean()*100; sig_dn = (res["code"][v] == -1).mean()*100
        print(f"[TREND] valid pixels {nv:,}; mean slope {sl[v].mean():+.5f}/yr")
        print(f"[TREND] increasing {inc:.1f}%  decreasing {dec:.1f}%  (significant p<{ALPHA}: up {sig_up:.1f}%, down {sig_dn:.1f}%)")
        print(f"[TREND] Hurst>0.5 {hp:.1f}%  persistent-decline {pdec:.1f}% (priority)")
    ref = os.path.join(out, "eqi_trend_core.npy" if mode == "core" else "eqi_trend.npy")
    if os.path.exists(ref) and os.path.exists(os.path.join(out, "eqi_yearly_mean.csv")):
        r = np.load(ref, mmap_mode="r"); both = v & np.isfinite(r)
        if both.any():
            d = np.abs(sl[both] - np.asarray(r)[both])
            print(f"[TREND] check vs 05 {os.path.basename(ref)}: {int(both.sum()):,} px, max |diff| {d.max():.2e}")
    if prof is not None:
        print("[TREND] wrote GeoTIFFs: trend_slope/pvalue/code/hurst/persist .tif")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        k = max(1, max(sl.shape) // 800)                       # decimate for the preview
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
        im0=ax[0].imshow(sl[::k, ::k], cmap="RdBu", vmin=-0.01, vmax=0.01); ax[0].set_title("Theil-Sen slope"); plt.colorbar(im0,ax=ax[0],shrink=.8)
        im1=ax[1].imshow(res["hurst"][::k, ::k], cmap="viridis", vmin=0, vmax=1); ax[1].set_title("Hurst"); plt.colorbar(im1,ax=ax[1],shrink=.8)
        im2=ax[2].imshow(res["persist"][::k, ::k], cmap="coolwarm", vmin=-2, vmax=2); ax[2].set_title("Persistence"); plt.colorbar(im2,ax=ax[2],shrink=.8)
        plt.tight_layout(); plt.savefig(os.path.join(out, "06_trend_preview.png"), dpi=120)
        print(f"[TREND] preview -> {out}/06_trend_preview.png")
    except Exception as e:
        print(f"[TREND] (preview skipped: {e})")
    if os.path.basename(path).startswith("_demo"):
        os.remove(path)
    print("[TREND] done.")


if __name__ == "__main__":
    main()
