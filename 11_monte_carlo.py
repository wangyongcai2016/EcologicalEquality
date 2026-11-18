#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 11 — Monte-Carlo uncertainty propagation
================================================================================
Paper 2.7/2.10: turns point statements ("30.2% of the area declined") into interval statements and shows which
subjective choice moves them. The full national stacks x N draws is not feasible, so the propagation runs on a
simple random sample of N_PIX analysed pixels (equal-area 500 m grid, so a pixel share is an area share); the
sampling error of that sample is reported separately (binomial SE, a lower bound because neighbours are correlated).

Uncertain inputs (every draw re-computes EFI, EPI and EQI from the pipeline's own stacks):
    * EFI reference percentile q ~ U(88, 92)   (01_efi_new uses 90): per-unit reference = mean of the pixel-mean
      values >= q-th percentile among stable pixels, recomputed from ALL stable pixels of each assessment unit
    * EPI blend lambda ~ U(0.4, 0.8)           (04_epi uses 0.6)
    * EPI human-pressure scale f ~ N(1, 0.15) clipped to [0.5, 1.5], ONE factor per draw applied to every pixel and
      year. It stands for the shared, systematic error of source weights / d_max / sensitivities. (Independent
      pixel noise, used before, averages out in area statistics and understates the uncertainty.) The coefficient
      table is not re-sampled and 04_epi is not re-run: this is an emulation, not a full propagation.
    * EQI weights ~ Dirichlet(WEIGHT_CONC * base weights); base = equal weights, or one of WEIGHT_SCHEMES drawn
      at random (eqi_weights.csv from 05_eqi.py)
NOT perturbed: ESI (STL / window / normalisation choices), the drought term itself, the stable-pixel mask,
the input products. State this as a limitation.

Draw 0 is the base case (q=90, lambda=LAMBDA of 04_epi, f=1, equal weights) and is checked against the pipeline's
own files (efi_stack, eqi_stack, eqi_trend_core, efi_summary.csv); a WARNING is printed if they disagree.

Headline statistics per draw (core years, normally 2002-2020): mean EQI, EQI change first->last year, mean
Theil-Sen slope, share of pixels with a negative slope, and the four dEFI/dESI regime shares of 08_diagnostic_space
(dEFI recomputed per draw on the EFI of START_YEAR..LU_END_YEAR; dESI = esi_trend.npy, fixed).

INPUT (OUT_DIR): efi_mean/esi/epi_total .npy, efi_stack/esi_stack/epi_stack .npy, esi_trend.npy, drought_pressure.npy,
                 eqi_stack.npy + eqi_yearly_mean.csv + eqi_trend_core.npy + eqi_weights.csv (05), efi_summary.csv (01),
                 the aligned FVC / LAI / GPP rasters (CONFIG), UNITS_TIF, STABLE_TIF
OUTPUT (OUT_DIR): mc_summary.csv  statistic, base, mc_mean, ci_low, ci_high, mc_sd, sampling_se
                  mc_draws.csv    parameters and statistics of every draw
                  mc_sensitivity.csv  Spearman rank correlation of each parameter with each statistic
                  mc_sample.csv   per sampled pixel: base EQI, MC mean and 95% interval, base slope, P(decline)
                  11_mc_preview.png
        _mc_cache/ holds the raster scan (pixel means + sampled series); it is reused while the rasters are
        unchanged and can be deleted.
Environment: EQI_MC_DRAWS (default 500), EQI_MC_PIX (default 100000).   Needs the real outputs; no synthetic demo.
================================================================================
"""
from __future__ import annotations
import os, re, csv, time, importlib.util, warnings
import numpy as np
import geo_io as gio
from geo_io import CONFIG

warnings.filterwarnings("ignore", category=RuntimeWarning)

N_DRAWS = int(os.environ.get("EQI_MC_DRAWS", 500))
N_PIX = int(os.environ.get("EQI_MC_PIX", 30_000))          # pixels PER ecosystem type (stratified)
SEED = 2024
REF_PERCENTILE = 90.0                 # must equal 01_efi_new.REF_PERCENTILE
MIN_REF_PIXELS = 30                   # must equal 01_efi_new.MIN_REF_PIXELS
Q_RANGE = (88.0, 92.0)
LAMBDA_RANGE = (0.4, 0.8)
HUMAN_SD, HUMAN_CLIP = 0.15, (0.5, 1.5)
WEIGHT_SCHEMES = ("equal",)           # pool of base weights: "equal", "entropy", "ahp_entropy"
WEIGHT_CONC = 200.0
ROBUST = 0.95                         # a pixel's trend sign is "robust" when it holds in >= 95% of the draws
PARAMS = ("fvc", "lai", "gpp")
TYPES = {1: "forest", 2: "shrub", 3: "grassland", 4: "wetland", 5: "sparse"}
REG = ("true_recovery", "fragile_greening", "low_but_stable", "collapse")
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- helpers
def _load_module(fname):
    spec = importlib.util.spec_from_file_location("m_" + fname[:2], os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _read_const(fname, name, default):
    try:
        txt = open(os.path.join(HERE, fname), encoding="utf-8").read()
        return float(re.search(rf"^{name}\s*=\s*([0-9.]+)", txt, re.M).group(1))
    except Exception:
        return default


def _ts(Y):
    """Theil-Sen slope per column of Y (T, n); NaN when a column has a NaN."""
    ii, jj = np.triu_indices(Y.shape[0], 1)
    S = (Y[jj] - Y[ii]) / (jj - ii).astype("float32")[:, None]
    return np.median(S, axis=0)


def _at(arr, idx, rows, cols):
    out = np.empty((len(idx), rows.size), "float32")
    for k, t in enumerate(idx):
        out[k] = np.asarray(arr[t][rows, cols])
    return out


# ----------------------------------------------------------------- raster scan (pixel means + sampled series)
def _veg_inputs(c):
    veg_dirs = c.get("VEG_DIRS") or c.get("VEG_DIR", ".")
    patterns = c.get("VEG_PATTERNS", c.get("VEG_PATTERN"))
    dirs = {"fvc": c.get("FVC_DIR") or veg_dirs, "lai": c.get("LAI_DIR") or veg_dirs, "gpp": c.get("GPP_DIR") or veg_dirs}
    names = {"fvc": c["FVC_NAME"], "lai": c["LAI_NAME"], "gpp": c["GPP_NAME"]}
    return dirs, names, patterns


def _signature(c):
    """File sizes + mtimes of every vegetation raster: the cache is valid only while they are unchanged."""
    import glob
    dirs, names, _ = _veg_inputs(c)
    sig = []
    for p in PARAMS:
        d = gio._resolve_var_dir(dirs[p], names[p])
        for f in sorted(glob.glob(os.path.join(d, "*.tif"))):
            st = os.stat(f); sig.append(f"{f}|{st.st_size}|{int(st.st_mtime)}")
    return "\n".join(sig)


def scan_rasters(c, out, rows, cols, years, tag):
    """Returns pm {param: (H,W) float32 time-mean of the pixel series (as 01_efi_new)}, X (3, len(trend years), n)."""
    from concurrent.futures import ThreadPoolExecutor
    cache = os.path.join(out, "_mc_cache"); os.makedirs(cache, exist_ok=True)
    sigp = os.path.join(cache, "signature.txt"); xp = os.path.join(cache, f"X_{tag}.npy")
    pmp = {p: os.path.join(cache, f"pm_{p}.npy") for p in PARAMS}
    sig = _signature(c)
    if (all(os.path.exists(v) for v in pmp.values()) and os.path.exists(xp) and os.path.exists(sigp)
            and open(sigp, encoding="utf-8").read() == sig):
        print(f"[MC] raster scan: reusing cache {cache}")
        return {p: np.load(pmp[p]) for p in PARAMS}, np.load(xp)
    dirs, names, patterns = _veg_inputs(c)
    ty = years["trend"]; all_years = list(range(c["START_YEAR"], c["END_YEAR"] + 1))
    X = np.full((3, len(ty), rows.size), np.nan, "float32"); pm = {}
    t0 = time.time()
    for pi, p in enumerate(PARAMS):
        def read_year(y, p=p):
            return gio.stack_annual(dirs[p], names[p], y, y, patterns)[0][0]
        s = k = None
        with ThreadPoolExecutor(max_workers=4) as ex:
            for y, a in zip(all_years, ex.map(read_year, all_years)):
                if s is None:
                    s = np.zeros(a.shape, "float64"); k = np.zeros(a.shape, "int16")
                ok = np.isfinite(a); s[ok] += a[ok]; k += ok
                if y in ty:
                    X[pi, ty.index(y)] = a[rows, cols]
        pm[p] = np.where(k > 0, s / np.maximum(k, 1), np.nan).astype("float32")
        print(f"[MC]   scanned {p}: {len(all_years)} years, {time.time() - t0:.0f}s", flush=True)
        del s, k
    for p in PARAMS:
        np.save(pmp[p], pm[p])
    np.save(xp, X)
    open(sigp, "w", encoding="utf-8").write(sig)
    return pm, X


# ----------------------------------------------------------------- EFI reference tables
def build_tables(pm, units, stable):
    """{unit: (sorted values float64, suffix sums)} reproducing 01_efi_new.unit_reference_bootstrap's sample:
    stable pixels of the unit, or all finite pixels of the unit when it has fewer than MIN_REF_PIXELS."""
    fin = np.isfinite(pm) & (units > 0)
    sidx = np.flatnonzero((fin & stable).ravel())
    u = units.ravel()[sidx]; v = pm.ravel()[sidx]
    order = np.lexsort((v, u)); u = u[order]; v = v[order]
    ids, start = np.unique(u, return_index=True)
    end = np.r_[start[1:], u.size]
    tab = {int(k): v[a:b] for k, a, b in zip(ids, start, end)}
    for k in np.unique(units[fin]):
        k = int(k)
        if tab.get(k) is None or tab[k].size < MIN_REF_PIXELS:
            vv = np.sort(pm[fin & (units == k)])
            if vv.size:
                tab[k] = vv
            else:
                tab.pop(k, None)
    out = {}
    for k, vv in tab.items():
        v64 = vv.astype("float64")
        out[k] = (v64, np.cumsum(v64[::-1])[::-1])
    return out


def ref_at(entry, q):
    """Mean of the values >= the q-th percentile (linear interpolation), as np.percentile + vals[vals >= thr]."""
    v, ss = entry; n = v.size
    pos = q / 100.0 * (n - 1); lo = int(np.floor(pos)); hi = min(lo + 1, n - 1)
    thr = v[lo] + (v[hi] - v[lo]) * (pos - lo)
    j = int(np.searchsorted(v, thr, side="left"))
    if j >= n:
        j = 0
    return ss[j] / (n - j)


# ----------------------------------------------------------------- one draw (runs in a worker)
def eval_draw(fixed, ref, lam, f, w, keep_slope):
    X, unit_i, esi_c, hn, dr, esi_tr, core_pos, typ, wp = fixed
    with np.errstate(all="ignore"):
        r = np.clip(X / ref.astype("float32")[:, unit_i][:, None, :], 0.0, 1.0)
        efi = (r[0] + r[1] + r[2]) / 3.0                                   # (T_trend, n)
        epi = lam * np.clip(hn * f, 0.0, 1.0) + (1.0 - lam) * dr[None]
        eqi = w[0] * efi[core_pos] + w[1] * esi_c + w[2] * (1.0 - epi)     # (T_core, n); NaN where ESI/EPI are missing
        slope = _ts(eqi)
        dE = _ts(efi)
        em = eqi.mean(0)
    v = np.isfinite(slope)
    vr = np.isfinite(dE) & np.isfinite(esi_tr)
    up_f, up_s = dE > 0, esi_tr > 0
    cls = {"true_recovery": vr & up_f & up_s, "fragile_greening": vr & up_f & ~up_s,
           "low_but_stable": vr & ~up_f & up_s, "collapse": vr & ~up_f & ~up_s}
    def wshare(cond, pop):                                                # stratified (area-weighted) share, %
        d = wp[pop].sum()
        return 100.0 * float(wp[cond & pop].sum()) / d if d > 0 else np.nan
    me = np.isfinite(em)
    st = dict(
        eqi_mean=float((wp[me] * em[me]).sum() / wp[me].sum()),
        eqi_change=float(np.nansum(wp * (eqi[-1] - eqi[0])) / wp[v].sum()),
        slope_mean=float((wp[v] * slope[v]).sum() / wp[v].sum()),
        decline_pct=wshare(slope < 0, v),
        n_valid=int(v.sum()),
    )
    for k in REG:
        st[k + "_pct"] = wshare(cls[k], vr)
    for t, tn in TYPES.items():
        sel = typ == t
        nv = (vr & sel).sum()
        for k in REG:
            st[f"{tn}_{k}_pct"] = 100.0 * float((cls[k] & sel).sum()) / nv if nv else np.nan
        vt = v & sel
        st[f"{tn}_decline_pct"] = 100.0 * float((slope[vt] < 0).mean()) if vt.any() else np.nan
    return st, em.astype("float32"), (v & (slope < 0)), v, (slope.astype("float32") if keep_slope else None)


# ----------------------------------------------------------------- main
STAT_KEYS = ("eqi_mean", "eqi_change", "slope_mean", "decline_pct", "true_recovery_pct", "fragile_greening_pct",
             "low_but_stable_pct", "collapse_pct") + tuple(f"{tn}_{k}_pct" for tn in TYPES.values() for k in REG)             + tuple(f"{tn}_decline_pct" for tn in TYPES.values())


def main():
    c = CONFIG; out = gio.ensure_outdir(); t_start = time.time()
    need = ["efi_mean.npy", "esi.npy", "epi_total.npy", "efi_stack.npy", "esi_stack.npy", "epi_stack.npy",
            "esi_trend.npy", "drought_pressure.npy", "eqi_stack.npy", "eqi_yearly_mean.csv", "eqi_trend_core.npy",
            "eqi_weights.csv", "efi_summary.csv"]
    miss = [n for n in need if not os.path.exists(os.path.join(out, n))]
    if miss:
        raise FileNotFoundError(f"[MC] missing in {out}: {miss}. Run 01-05 first (11 has no synthetic demo).")
    m05 = _load_module("05_eqi.py")
    arrs, years = m05.stack_years(out, c)
    if arrs is None:
        raise FileNotFoundError(f"[MC] {years}")
    core = sorted(set(years["efi"]) & set(years["esi"]) & set(years["epi"]))
    ty = list(range(c["START_YEAR"], min(c["END_YEAR"], c.get("LU_END_YEAR", c["END_YEAR"])) + 1))
    if not set(core) <= set(ty):
        raise ValueError(f"[MC] core years {core[0]}-{core[-1]} not inside the EFI trend years {ty[0]}-{ty[-1]}")
    core_pos = np.array([ty.index(y) for y in core])
    yrs_all = dict(years, trend=ty)
    print(f"[MC] draws {N_DRAWS} (+ base), sample {N_PIX:,} px; core years {core[0]}-{core[-1]}, "
          f"EFI trend years {ty[0]}-{ty[-1]}", flush=True)
    lam0 = _read_const("04_epi.py", "LAMBDA", 0.6)

    # ---- sample of analysed pixels
    rng = np.random.default_rng(SEED)
    mask = gio.read_mask(); units = gio.read_units()
    efi_m = np.load(os.path.join(out, "efi_mean.npy")); esi_m = np.load(os.path.join(out, "esi.npy"))
    epi_m = np.load(os.path.join(out, "epi_total.npy"))
    esi_tr_full = np.load(os.path.join(out, "esi_trend.npy"), mmap_mode="r")
    # population of the regime statistics = pixels with a finite dEFI and dESI (the 08 population); ESI / EPI series may
    # have gaps (their EQI is then NaN and they only enter the regime shares, not the EQI trend statistics)
    cand = mask & np.isfinite(efi_m) & np.isfinite(np.asarray(esi_tr_full))
    del efi_m, esi_m, epi_m
    etype_full = np.nan_to_num(gio.read_tif(os.path.join(out, "eco_type.tif"))[0], nan=0).astype("uint8")
    H, W = mask.shape
    picks, ptype, ntot = [], [], {}
    for t in range(0, 6):                                   # stratified by ecosystem type (0 = unclassified)
        ft = np.flatnonzero((cand & (etype_full == t)).ravel())
        if ft.size == 0:
            continue
        ntot[t] = int(ft.size)
        k = min(int(N_PIX * 1.12) + 1000, ft.size)
        picks.append(np.sort(rng.choice(ft, size=k, replace=False))); ptype.append(np.full(k, t, "uint8"))
    pick = np.concatenate(picks); typ_over = np.concatenate(ptype)
    n_over = pick.size; n_cand = sum(ntot.values())
    rows, cols = np.divmod(pick, W)
    print(f"[MC] candidate pixels {n_cand:,} in {len(ntot)} strata {ntot}; drawn {n_over:,} before the completeness filter", flush=True)

    # ---- raster scan: pixel-mean fields and sampled FVC/LAI/GPP series
    import hashlib
    tag = hashlib.md5(rows.tobytes() + cols.tobytes() + str(ty).encode()).hexdigest()[:12]   # cache key = pixels + years
    pm, X = scan_rasters(c, out, rows, cols, yrs_all, tag)

    # ---- sampled series from the pipeline stacks
    esi_c = _at(arrs["esi"], [years["esi"].index(y) for y in core], rows, cols)
    epi_c = _at(arrs["epi"], [years["epi"].index(y) for y in core], rows, cols)
    dr = np.clip(np.nan_to_num(np.asarray(np.load(os.path.join(out, "drought_pressure.npy"), mmap_mode="r")[rows, cols],
                                          "float32"), nan=0.0), 0, 1)
    esi_tr = np.asarray(np.load(os.path.join(out, "esi_trend.npy"), mmap_mode="r")[rows, cols], "float32")
    hn = np.clip((epi_c - (1.0 - lam0) * dr[None]) / lam0, 0.0, 1.0)           # normalised human pressure, recovered
    unit_s = units[rows, cols]

    tabs = {p: build_tables(pm[p], units, mask) for p in PARAMS}
    uu = np.array(sorted(set.intersection(*[set(t) for t in tabs.values()])))
    has_ref = np.isin(unit_s, uu)
    keep = np.isfinite(X).all((0, 1)) & has_ref & np.isfinite(esi_tr)
    kidx = np.flatnonzero(keep)
    sel = []
    for t in sorted(ntot):                                  # at most N_PIX per stratum
        it = kidx[typ_over[kidx] == t]
        sel.append(np.sort(rng.choice(it, size=N_PIX, replace=False)) if it.size > N_PIX else it)
    kidx = np.concatenate(sel)
    typ = typ_over[kidx]
    nkept = {t: int((typ == t).sum()) for t in sorted(ntot)}
    wt_t = {t: ntot[t] / max(nkept[t], 1) for t in ntot}      # pixels represented by one sampled pixel of the stratum
    wp = np.array([wt_t[t] for t in typ], "float64")
    complete = np.isfinite(esi_c[:, kidx]).all(0) & np.isfinite(epi_c[:, kidx]).all(0)
    share_complete = 100.0 * float((wp * complete).sum() / wp.sum())
    print(f"[MC] kept per stratum {nkept}; area-weighted share with a complete ESI/EPI core series {share_complete:.1f}%", flush=True)
    rows, cols, unit_s = rows[kidx], cols[kidx], unit_s[kidx]
    X, esi_c, epi_c, hn, dr, esi_tr = X[:, :, kidx], esi_c[:, kidx], epi_c[:, kidx], hn[:, kidx], dr[kidx], esi_tr[kidx]
    n = kidx.size
    unit_i = np.searchsorted(uu, unit_s)
    fixed = (X, unit_i, esi_c, hn, dr, esi_tr, core_pos, typ, wp)

    # ---- draws (index 0 = base case)
    wtab = {}
    with open(os.path.join(out, "eqi_weights.csv"), encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            wtab[r["scheme"]] = np.array([float(r["W_EFI"]), float(r["W_ESI"]), float(r["W_EPI"])])
    wtab["equal"] = np.full(3, 1 / 3)
    for s in WEIGHT_SCHEMES:
        if s not in wtab:
            raise KeyError(f"[MC] weighting scheme '{s}' not in eqi_weights.csv ({list(wtab)})")
    D = N_DRAWS
    q = np.r_[REF_PERCENTILE, rng.uniform(*Q_RANGE, D)]
    lam = np.r_[lam0, rng.uniform(*LAMBDA_RANGE, D)]
    fh_ = np.r_[1.0, np.clip(rng.normal(1.0, HUMAN_SD, D), *HUMAN_CLIP)]
    sch = rng.integers(len(WEIGHT_SCHEMES), size=D)
    Wd = np.vstack([np.full(3, 1 / 3)] + [rng.dirichlet(WEIGHT_CONC * wtab[WEIGHT_SCHEMES[s]]) for s in sch])
    R = np.empty((D + 1, 3, uu.size))
    for d in range(D + 1):
        for pi, p in enumerate(PARAMS):
            R[d, pi] = [ref_at(tabs[p][int(u)], q[d]) for u in uu]

    # ---- consistency of draw 0 with the pipeline
    warn = []
    es = {}
    with open(os.path.join(out, "efi_summary.csv"), encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            es[(int(r["unit"]), r["param"])] = float(r["ref"])
    dref = max(abs(es[(int(u), p)] - R[0, pi, j]) for pi, p in enumerate(PARAMS) for j, u in enumerate(uu)
               if (int(u), p) in es)
    efi_stack_s = _at(arrs["efi"], [years["efi"].index(y) for y in ty], rows, cols)
    st0, em0, _, _, slope0 = eval_draw(fixed, R[0], lam[0], 1.0, Wd[0], True)
    with np.errstate(all="ignore"):
        r0 = np.clip(X / R[0].astype("float32")[:, unit_i][:, None, :], 0, 1)
    defi = float(np.nanmax(np.abs((r0[0] + r0[1] + r0[2]) / 3.0 - efi_stack_s)))
    eqi_ref = np.asarray(np.load(os.path.join(out, "eqi_stack.npy"), mmap_mode="r"))  # only to index below
    ypipe, _ = gio.eqi_annual_years(out)
    eqi_pipe = _at(eqi_ref, [list(ypipe).index(y) for y in core], rows, cols)
    eqi0 = (Wd[0][0] * np.clip(r0, 0, 1).mean(0)[core_pos] + Wd[0][1] * esi_c +
            Wd[0][2] * (1 - (lam0 * hn + (1 - lam0) * dr[None])))
    deqi = float(np.nanmax(np.abs(eqi0 - eqi_pipe)))
    slope_pipe = np.asarray(np.load(os.path.join(out, "eqi_trend_core.npy"), mmap_mode="r")[rows, cols], "float32")
    dsl = float(np.nanmax(np.abs(slope0 - slope_pipe)))
    print(f"[MC] draw 0 vs pipeline: reference max|d|={dref:.1e} (csv has 4 decimals), EFI {defi:.1e}, "
          f"EQI {deqi:.1e}, slope {dsl:.1e}", flush=True)
    # efi_summary.csv keeps 4 decimals and 01 averages in float32: allow 1e-4 absolute + 1e-5 relative on the references
    ref_tol = max(1e-4 + 1e-5 * float(np.abs(R[0]).max()), 1e-4)
    if dref > ref_tol or defi > 1e-4 or deqi > 1e-4 or dsl > 1e-5:
        print("[MC] WARNING: draw 0 does not reproduce the pipeline files; check the CONFIG / rasters before using "
              "the intervals")

    # ---- run
    from joblib import Parallel, delayed
    nj = max(1, min(CONFIG.get("N_JOBS", 8), 24, D + 1))
    t0 = time.time()
    res = Parallel(n_jobs=nj)(delayed(eval_draw)(fixed, R[d], lam[d], fh_[d], Wd[d], d == 0) for d in range(D + 1))
    print(f"[MC] {D + 1} draws on {n:,} px, {nj} workers: {time.time() - t0:.0f}s", flush=True)
    stats = {k: np.array([r[0][k] for r in res]) for k in STAT_KEYS}
    base = {k: stats[k][0] for k in STAT_KEYS}
    mc = {k: stats[k][1:] for k in STAT_KEYS}
    EM = np.stack([r[1] for r in res[1:]])
    dec = np.sum([r[2] for r in res[1:]], axis=0); vcnt = np.sum([r[3] for r in res[1:]], axis=0)
    pdec = np.where(vcnt > 0, dec / np.maximum(vcnt, 1), np.nan)
    full = vcnt == D
    wv = wp * np.isfinite(pdec)
    rob_dn = float((wp * (full & (pdec >= ROBUST))).sum() / wv.sum() * 100)
    rob_up = float((wp * (full & (pdec <= 1 - ROBUST))).sum() / wv.sum() * 100)
    unc = 100.0 - rob_dn - rob_up
    lo_em, hi_em = np.nanpercentile(EM, [2.5, 97.5], axis=0); mean_em = np.nanmean(EM, axis=0)

    # ---- outputs
    Wt = {t: ntot[t] / sum(ntot.values()) for t in ntot}

    def se_share(p_pct, key=None):
        """Binomial sampling SE (a lower bound: neighbouring pixels are correlated). Per-type keys use that stratum's
        n; national keys combine the strata: sqrt(sum_t W_t^2 p_t (1-p_t) / n_t)."""
        for t, tn in TYPES.items():
            if key and key.startswith(tn + "_"):
                p = p_pct / 100.0
                return 100.0 * np.sqrt(max(p * (1 - p), 0) / max(nkept.get(t, 1), 1))
        kk = key.replace("_pct", "") if key else None
        tot = 0.0
        for t in ntot:
            pt = base.get(f"{TYPES[t]}_{kk}_pct") if (t in TYPES and kk) else None
            pt = (pt / 100.0) if (pt is not None and np.isfinite(pt)) else p_pct / 100.0
            tot += Wt[t] ** 2 * pt * (1 - pt) / max(nkept[t], 1)
        return 100.0 * np.sqrt(tot)
    rows_out = []
    labels = {"eqi_mean": f"EQI_mean_{core[0]}-{core[-1]}", "eqi_change": f"EQI_change_{core[0]}_to_{core[-1]}",
              "slope_mean": "EQI_mean_slope_per_yr", "decline_pct": "decline_share_pct",
              "true_recovery_pct": "true_recovery_pct", "fragile_greening_pct": "fragile_greening_pct",
              "low_but_stable_pct": "low_but_stable_pct", "collapse_pct": "collapse_pct"}
    labels.update({k: k for k in STAT_KEYS if k not in labels})
    for k in STAT_KEYS:
        a = mc[k]
        if k.endswith("_pct"):
            sse = se_share(base[k], k)
        elif k == "eqi_mean":
            sse = float(np.sqrt(sum(Wt[t] ** 2 * np.nanvar(em0[typ == t]) / max(nkept[t], 1) for t in ntot)))
        elif k == "slope_mean":
            sse = float(np.sqrt(sum(Wt[t] ** 2 * np.nanvar(slope0[typ == t]) / max(nkept[t], 1) for t in ntot)))
        else:
            sse = float("nan")
        rows_out.append([labels[k], f"{base[k]:.5f}", f"{np.nanmean(a):.5f}", f"{np.nanpercentile(a, 2.5):.5f}",
                         f"{np.nanpercentile(a, 97.5):.5f}", f"{np.nanstd(a):.5f}",
                         "" if not np.isfinite(sse) else f"{sse:.5f}"])
    # weighting scenarios: the base case (q=90, lambda of 04_epi, f=1) under each whole-period weighting scheme
    scen = {}
    for nm in wtab:
        if nm == "equal":
            continue
        s_nm = eval_draw(fixed, R[0], lam[0], 1.0, wtab[nm], False)[0]
        scen[nm] = s_nm
        for k in STAT_KEYS:
            rows_out.append([f"scenario_{nm}:{labels[k]}", f"{s_nm[k]:.5f}", "", "", "", "", ""])
    rows_out += [["complete_ESI_EPI_series_share_pct (area-weighted)", f"{share_complete:.2f}", "", "", "", "", ""],
                 ["robust_decline_pct(pixels, >=95% of draws)", f"{rob_dn:.3f}", "", "", "", "", ""],
                 ["robust_increase_pct", f"{rob_up:.3f}", "", "", "", "", ""],
                 ["uncertain_sign_pct", f"{unc:.3f}", "", "", "", "", ""],
                 ["n_sample_pixels", str(n), "", "", "", "", ""]]
    with open(os.path.join(out, "mc_summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["statistic", "base", "mc_mean", "ci_low", "ci_high", "mc_sd", "sampling_se"])
        w.writerows(rows_out)
    with open(os.path.join(out, "mc_draws.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["draw", "q", "lambda", "human_scale", "w_efi", "w_esi", "w_epi", *STAT_KEYS])
        for d in range(D + 1):
            w.writerow([d, f"{q[d]:.3f}", f"{lam[d]:.4f}", f"{fh_[d]:.4f}", *[f"{x:.4f}" for x in Wd[d]],
                        *[f"{stats[k][d]:.5f}" for k in STAT_KEYS]])
    from scipy.stats import spearmanr
    par = {"q": q[1:], "lambda": lam[1:], "human_scale": fh_[1:], "w_efi": Wd[1:, 0], "w_esi": Wd[1:, 1], "w_epi": Wd[1:, 2]}
    with open(os.path.join(out, "mc_sensitivity.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["parameter", *[labels[k] for k in STAT_KEYS]])
        for pn, pv in par.items():
            w.writerow([pn, *[f"{spearmanr(pv, mc[k], nan_policy='omit').correlation:+.3f}" for k in STAT_KEYS]])
    with open(os.path.join(out, "mc_sample.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["row", "col", "unit", "type", "weight", "eqi_base", "eqi_mc_mean", "eqi_ci_low", "eqi_ci_high",
                                         "slope_base", "p_decline"])
        for i in range(n):
            w.writerow([int(rows[i]), int(cols[i]), int(unit_s[i]), int(typ[i]), f"{wp[i]:.2f}", f"{em0[i]:.4f}", f"{mean_em[i]:.4f}",
                        f"{lo_em[i]:.4f}", f"{hi_em[i]:.4f}", f"{slope0[i]:.6f}", f"{pdec[i]:.3f}"])

    print("[MC] statistic                          base      MC mean   95% interval            sampling SE")
    for r in rows_out[:8]:
        print(f"      {r[0]:34s} {float(r[1]):9.4f} {float(r[2]):9.4f}  [{float(r[3]):.4f}, {float(r[4]):.4f}]  {r[6]}")
    print(f"[MC] pixels whose trend sign holds in >= {ROBUST:.0%} of draws: decline {rob_dn:.1f}%, increase {rob_up:.1f}%, "
          f"uncertain {unc:.1f}%")
    print("[MC] Spearman (param vs statistic) in mc_sensitivity.csv")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        ax[0].hist(mc["decline_pct"], bins=30, color="#4c72b0"); ax[0].axvline(base["decline_pct"], color="k")
        ax[0].set_title("Decline share (%)"); ax[0].set_xlabel("% pixels with negative EQI slope")
        ax[1].hist(mc["eqi_mean"], bins=30, color="#55a868"); ax[1].axvline(base["eqi_mean"], color="k")
        ax[1].set_title("Mean EQI"); ax[1].set_xlabel("EQI")
        sc = ax[2].scatter(cols, -rows, c=pdec, s=0.4, cmap="RdBu_r", vmin=0, vmax=1)
        ax[2].set_title("P(decline) over draws"); ax[2].set_aspect("equal"); plt.colorbar(sc, ax=ax[2], shrink=.8)
        plt.tight_layout(); plt.savefig(os.path.join(out, "11_mc_preview.png"), dpi=120)
        print(f"[MC] preview -> {out}/11_mc_preview.png")
    except Exception as e:
        print(f"[MC] (preview skipped: {e})")
    print(f"[MC] done in {time.time() - t_start:.0f}s.")


if __name__ == "__main__":
    main()
